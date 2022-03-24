import multiprocessing as mp

mp.set_start_method('spawn', force=True)
import os, copy
import argparse
import json
import pprint
import socket
import time
import random
from easydict import EasyDict
import yaml
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from calc_mAP import run_evaluation
from datasets import ava, spatial_transforms, temporal_transforms
from distributed_utils import init_distributed
import losses
from models import AVA_model
from scheduler import get_scheduler
from utils import *


def main(local_rank, args):
    '''dist init'''
    rank, world_size = init_distributed(local_rank, args)

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    opt = EasyDict(config)
    opt.world_size = world_size

    if rank == 0:
        mkdir(opt.result_path)
        mkdir(os.path.join(opt.result_path, 'tmp'))
        with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file, indent=2)
        logger = create_logger(os.path.join(opt.result_path, 'log.txt'))
        logger.info('opt: {}'.format(pprint.pformat(opt, indent=2)))

        writer = SummaryWriter(os.path.join(opt.result_path, 'tb'))
    else:
        logger = writer = None
    dist.barrier()

    random_seed(opt.manual_seed)
    # setting benchmark to True causes OOM in some cases
    if opt.get('cudnn', None) is not None:
        torch.backends.cudnn.deterministic = opt.cudnn.get('deterministic', False)
        torch.backends.cudnn.benchmark = opt.cudnn.get('benchmark', False)

    # create model
    net = AVA_model(opt.model)
    net.cuda()
    net = DistributedDataParallel(net, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)

    if rank == 0:
        logger.info(net)
        logger.info(parameters_string(net))

    if not opt.get('evaluate', False):
        train_aug = opt.train.augmentation

        spatial_transform = [getattr(spatial_transforms, aug.type)(**aug.get('kwargs', {})) for aug in
                             train_aug.spatial]
        spatial_transform = spatial_transforms.Compose(spatial_transform)

        temporal_transform = getattr(temporal_transforms, train_aug.temporal.type)(
            **train_aug.temporal.get('kwargs', {}))

        train_data = ava.AVA(
            opt.train.root_path,
            opt.train.annotation_path,
            spatial_transform,
            temporal_transform
        )

        train_sampler = DistributedSampler(train_data, round_down=True)

        train_loader = ava.AVADataLoader(
            train_data,
            batch_size=opt.train.batch_size,
            shuffle=False,
            num_workers=opt.train.get('workers', 1),
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True
        )

        if rank == 0:
            logger.info('# train data: {}'.format(len(train_data)))
            logger.info('train spatial aug: {}'.format(spatial_transform))
            logger.info('train temporal aug: {}'.format(temporal_transform))

            train_logger = Logger(
                os.path.join(opt.result_path, 'train.log'),
                ['epoch', 'loss', 'lr'])
            train_batch_logger = Logger(
                os.path.join(opt.result_path, 'train_batch.log'),
                ['epoch', 'batch', 'iter', 'loss', 'lr'])
        else:
            train_logger = train_batch_logger = None

        optim_opt = opt.train.optimizer
        sched_opt = opt.train.scheduler

        optimizer = getattr(optim, optim_opt.type)(
            net.parameters(),
            lr=sched_opt.base_lr,
            **optim_opt.kwargs
        )
        scheduler = get_scheduler(sched_opt, optimizer, opt.train.n_epochs, len(train_loader))

    val_aug = opt.val.augmentation

    transform_choices, total_choices = [], 1
    for aug in val_aug.spatial:
        kwargs_list = aug.get('kwargs', {})
        if not isinstance(kwargs_list, list):
            kwargs_list = [kwargs_list]
        cur_choices = [getattr(spatial_transforms, aug.type)(**kwargs) for kwargs in kwargs_list]
        transform_choices.append(cur_choices)
        total_choices *= len(cur_choices)

    spatial_transform = []
    for choice_idx in range(total_choices):
        idx, transform = choice_idx, []
        for cur_choices in transform_choices:
            n_choices = len(cur_choices)
            cur_idx = idx % n_choices
            transform.append(cur_choices[cur_idx])
            idx = idx // n_choices
        spatial_transform.append(spatial_transforms.Compose(transform))

    temporal_transform = getattr(temporal_transforms, val_aug.temporal.type)(**val_aug.temporal.get('kwargs', {}))

    val_data = ava.AVAmulticrop(
        opt.val.root_path,
        opt.val.annotation_path,
        spatial_transform,
        temporal_transform
    )

    val_sampler = DistributedSampler(val_data, round_down=False)

    val_loader = ava.AVAmulticropDataLoader(
        val_data,
        batch_size=opt.val.batch_size,
        shuffle=False,
        num_workers=opt.val.get('workers', 1),
        pin_memory=True,
        sampler=val_sampler
    )

    val_logger = None
    if rank == 0:
        logger.info('# val data: {}'.format(len(val_data)))
        logger.info('val spatial aug: {}'.format(spatial_transform))
        logger.info('val temporal aug: {}'.format(temporal_transform))

        val_log_items = ['epoch']
        if opt.val.with_label:
            val_log_items.append('loss')
        if opt.val.get('eval_mAP', None) is not None:
            val_log_items.append('mAP')
        if len(val_log_items) > 1:
            val_logger = Logger(
                os.path.join(opt.result_path, 'val.log'),
                val_log_items)

    if opt.get('pretrain', None) is not None:
        load_pretrain(opt.pretrain, net)

    begin_epoch = 1
    if opt.get('resume_path', None) is not None:
        if not os.path.isfile(opt.resume_path):
            opt.resume_path = os.path.join(opt.result_path, opt.resume_path)
        checkpoint = torch.load(opt.resume_path, map_location=lambda storage, loc: storage.cuda())

        begin_epoch = checkpoint['epoch'] + 1
        # begin_epoch = 6
        net.load_state_dict(checkpoint['state_dict'])
        if rank == 0:
            logger.info('Resumed from checkpoint {}'.format(opt.resume_path))

        if not opt.get('evaluate', False):
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if rank == 0:
                logger.info('Also loaded optimizer and scheduler from checkpoint {}'.format(opt.resume_path))

    criterion, act_func = getattr(losses, opt.loss.type)(**opt.loss.get('kwargs', {}))
    alpha1 = torch.tensor(0.5).float().cuda()
    alpha2 = torch.tensor(0.5).float().cuda()
    if opt.get('evaluate', False):  # evaluation mode
        val_epoch(begin_epoch - 1, val_loader, net, criterion, act_func,
                  opt, logger, val_logger, rank, world_size, writer)
    else:  # training and validation mode
        for e in range(begin_epoch, opt.train.n_epochs + 1):
            train_sampler.set_epoch(e)
            train_epoch(e, train_loader, net, alpha1, alpha2, criterion, optimizer, scheduler,
                        opt, logger, train_logger, train_batch_logger, rank, world_size, writer)

            if e % opt.train.val_freq == 0:
                val_epoch(e, val_loader, net, criterion, act_func,
                          opt, logger, val_logger, rank, world_size, writer)

    if rank == 0:
        writer.close()


def cc(x, y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


def multiple_cc(num_rois_ps, num_rois_obj, bg_feats, ps_feats, obj_feats):
    a, b, c = bg_feats.shape

    bg_feats_pb = bg_feats.unsqueeze(0).repeat((num_rois_ps, 1, 1, 1))
    ps_feats_pb = ps_feats.unsqueeze(2).unsqueeze(2).expand_as(bg_feats_pb).reshape(num_rois_ps * a * b * c)
    bg_feats_pb = bg_feats_pb.reshape(num_rois_ps * a * b * c)
    r_ps_bg = cc(bg_feats_pb, ps_feats_pb)

    if obj_feats is not None:
        bg_feats_ob = bg_feats.unsqueeze(0).repeat((num_rois_obj, 1, 1, 1))
        obj_feats_ob = obj_feats.unsqueeze(2).unsqueeze(2).expand_as(bg_feats_ob).reshape(num_rois_obj * a * b * c)
        bg_feats_ob = bg_feats_ob.reshape(num_rois_obj * a * b * c)
        r_obj_bg = cc(bg_feats_ob, obj_feats_ob)

        lcm = int(num_rois_ps * num_rois_obj / math.gcd(num_rois_ps, num_rois_obj))
        obj_feats_op = obj_feats.repeat((int(lcm / num_rois_obj), 1)).reshape(-1)
        ps_feats_op = ps_feats.repeat(int(lcm / num_rois_ps), 1).reshape(-1)
        r_ps_obj = cc(ps_feats_op, obj_feats_op)

        r = torch.sqrt(
            torch.abs(r_ps_bg ** 2 + r_obj_bg ** 2 - 2 * r_ps_obj * r_obj_bg * r_ps_obj) / (1 - r_ps_obj ** 2))
    else:
        r = r_ps_bg
    return torch.clamp(r, min=0.0, max=1.0)


def discorr(reduce_bg_feats, outputs, model):
    a, b, c = reduce_bg_feats.shape
    repeat_outputs = outputs.unsqueeze(0).repeat((a, 1, 1)).reshape(a * b, -1)
    reduce_bg_feats = reduce_bg_feats.reshape(a * b, -1)
    outputs_distM = torch.cdist(repeat_outputs, repeat_outputs)
    bg_feats_distM = torch.cdist(reduce_bg_feats, reduce_bg_feats)

    dcdist_outputs = outputs_distM - torch.mean(outputs_distM, 0) - torch.mean(outputs_distM, 1) + torch.mean(
        outputs_distM)
    dcdist_bg_feats = bg_feats_distM - torch.mean(bg_feats_distM, 0) - torch.mean(bg_feats_distM, 1) + torch.mean(
        bg_feats_distM)
    dcov_ob = torch.sum(dcdist_outputs * dcdist_bg_feats) / float(a * b * a * b)
    dcov_oo = torch.sum(dcdist_outputs * dcdist_outputs) / float(a * b * a * b)
    dcov_bb = torch.sum(dcdist_bg_feats * dcdist_bg_feats) / float(a * b * a * b)
    if (dcov_ob <= 0) or (dcov_oo <= 0) or (dcov_ob <= 0):
        dist_r = torch.tensor(0).float().cuda()
        for param in model.parameters():
            if param.requires_grad:
                dist_r = dist_r + param.sum()
        dist_r = 0. * dist_r
    else:
        dist_r = torch.sqrt(dcov_ob) / torch.sqrt(torch.sqrt(dcov_bb) * torch.sqrt(dcov_oo))
    return torch.clamp(dist_r, min=0.0, max=1.0)


def kernel(X, sigma):
    X = X.view(len(X), -1)
    XX = X @ X.t()
    X_sqnorms = torch.diag(XX)
    X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
    gamma = 1 / (2 * sigma ** 2)
    kernel_XX = torch.exp(-gamma * X_L2)
    return kernel_XX


def hsic_loss(input1, input2, unbiased=False, alternative=True):
    N = len(input1)
    if N < 4:
        return torch.tensor(0.0).to(input1.device)
    # we simply use the squared dimension of feature as the sigma for RBF kernel
    sigma_x = np.sqrt(input1.size()[1])
    sigma_y = np.sqrt(input2.size()[1])

    # compute the kernels
    kernel_XX = kernel(input1, sigma_x)
    kernel_YY = kernel(input2, sigma_y)

    if unbiased:
        """
        Unbiased estimator of Hilbert-Schmidt Independence Criterion
        Song, Le, et al. "Feature selection via dependence maximization." 2012.
        """
        tK = kernel_XX - torch.diag(kernel_XX)
        tL = kernel_YY - torch.diag(kernel_YY)
        hsic = (
                torch.trace(tK @ tL)
                + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
        )
        if alternative:
            loss = hsic
        else:
            loss = hsic / (N * (N - 3))
    else:
        """
        Biased estimator of Hilbert-Schmidt Independence Criterion
        Gretton, Arthur, et al. "Measuring statistical dependence with Hilbert-Schmidt norms." 2005.
        """
        KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
        LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
        loss = torch.trace(KH @ LH / (N - 1) ** 2)
    return loss


def kl_divergence(alpha, num_classes, device=None):
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def kl_loss(outputs, targets):
    evidence = F.relu(outputs)
    alpha = evidence + 1
    kl_alpha = (alpha - 1) * (1 - targets) + 1
    kl_loss = torch.mean(kl_divergence(kl_alpha, targets.shape[1], device=outputs.device))
    return kl_loss


def mean(list):
    return sum(list).to(dtype=torch.float) / len(list)


def train_epoch(epoch, data_loader, model, alpha1, alpha2, criterion, optimizer, scheduler,
                opt, logger, epoch_logger, batch_logger, rank, world_size, writer):
    if rank == 0:
        logger.info('Training at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter(opt.print_freq)
    data_time = AverageMeter(opt.print_freq)
    loss_time = AverageMeter(opt.print_freq)
    losses = AverageMeter(opt.print_freq)
    global_losses = AverageMeter()

    eps1, eps2 = torch.tensor(opt.train.eps1).float().cuda(), torch.tensor(opt.train.eps2).float().cuda()

    end_time = time.time()
    for i, data in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        curr_step = (epoch - 1) * len(data_loader) + i
        scheduler.step(curr_step)

        buffer = list()
        for k in range(opt.train.k_iter):

            ret = model(data)

            # num_rois = ret['num_rois']
            num_rois_ps = ret['num_rois_ps']
            num_rois_obj = ret['num_rois_obj']
            bg_feats = ret['bg_feats']
            ps_feats = ret['ps_feats']
            obj_feats = ret['obj_feats']
            reduce_bg_feats = ret['reduce_bg_feats']
            outputs = ret['outputs']
            targets = ret['targets']
            B_alpha = ret['B_alpha']
            B_beta = ret['B_beta']

            # if num_rois_ps > 0:
            #     hsic = hsic_loss(ps_feats, ps_feats, unbiased=True, alternative=True)
            #     kl = kl_loss(outputs, targets)
            # r = multiple_cc(num_rois_ps, num_rois_obj, bg_feats, ps_feats, obj_feats)
            # dist_r = discorr(reduce_bg_feats, outputs, model)

            # print('=======================================================')
            # print("k:", k + 1)
            # print("eps1:", eps1)
            # print("eps2:", eps2)
            # print("alpha1:", alpha1)
            # print("alpha2:", alpha2)
            # print("outputs:", outputs)
            # print("targets:", targets.shape)
            # print("B_alpha", torch.min(B_alpha))
            # print("B_beta", torch.min(B_beta))
            # print("reduce_bg_feats", reduce_bg_feats.shape)
            # print("bg_feats:", ret['bg_feats'].shape)

            # print('bg_feats_norm:', torch.max(torch.norm(bg_feats, dim=[1, 2])))

            # print("ps_feats:", ret['ps_feats'])
            # try:
            #     print("obj_feats:", ret['obj_feats'].shape)
            # except:
            #     print("obj_feats:", None)
            # print("r:", r)
            # print("dist_r:", dist_r)
            # print('=======================================================')

            tot_rois = torch.Tensor([num_rois_ps]).cuda()
            dist.all_reduce(tot_rois)
            tot_rois = tot_rois.item()

            if tot_rois == 0:
                end_time = time.time()
                continue

            # print(tot_rois)

            optimizer.zero_grad()

            if num_rois_ps > 0:
                loss = criterion(B_alpha, B_beta, targets)
                # print(alpha1, alpha2)
                L = loss
                    # + alpha2 * (hsic - eps2)
                # alpha1 * (eps1 - r) + alpha2 * (dist_r - eps2)
                # loss = loss * num_rois_ps / tot_rois * world_size
                L = L * num_rois_ps / tot_rois * world_size
            else:
                loss = torch.tensor(0).float().cuda()
                for param in model.parameters():
                    if param.requires_grad:
                        loss = loss + param.sum()
                loss = 0. * loss
                L = loss

            L.backward()

            # grads = [torch.max(param.grad.view(-1)).item() for param in model.parameters()]
            # print("grads", grads)

            # value = random.uniform(0.04, 0.1)
            value = 0.1
            nn.utils.clip_grad_value_(model.parameters(), clip_value=value)
            # grads = [torch.max(param.grad.view(-1)).item() for param in model.parameters()]
            # print("grads", grads)

            optimizer.step()

            # temp_model_paras = [param.clone() for param in model.parameters()]
            # buffer.append(temp_model_paras)
            # param_tilde = (*map(mean, zip(*buffer))),
            # for idx, param in enumerate(model.parameters()):
            #     param.data = param_tilde[idx]

            # if num_rois_ps > 0:
            #     update_ret = model(data)
            #     update_bg_feats = update_ret['bg_feats']
            #     update_ps_feats = update_ret['ps_feats']
            #     update_obj_feats = update_ret['obj_feats']
            #     update_reduce_bg_feats = update_ret['reduce_bg_feats']
            #     update_outputs = update_ret['outputs']
            #     update_r = multiple_cc(num_rois_ps, num_rois_obj, update_bg_feats, update_ps_feats, update_obj_feats)
            # update_dist_r = discorr(update_reduce_bg_feats, update_outputs, model)

            # print('=======================================================')
            # print("update_bg_feats", update_bg_feats)

            # print("update_bg_feats_norm:", torch.max(torch.norm(update_bg_feats, dim=[1, 2])))

            # print("update_ps_feats", update_ps_feats)
            # print("update_obj_feats", update_obj_feats)
            # print("update_reduce_bg_feats", update_reduce_bg_feats)
            # print("update_outputs", update_outputs)
            # print("update_r:", update_r)
            # print("update_dist_r:", update_dist_r)
            # print('=======================================================')

            # print("after loss---------------------------------------->", criterion(update_outputs, targets))

            # alpha1 += opt.train.scheduler.dual_lr1 * (eps1 - update_r - opt.train.scheduler.delta1 * alpha1)
            # # alpha1 = torch.clamp(alpha1, min=0.0)
            # # print(alpha1)
            # alpha1 = alpha1.detach()
            # if alpha1.item() < 0.:
            #     alpha1 = torch.tensor(0).float().cuda()
            # # print(alpha1)
            # alpha2 += opt.train.scheduler.dual_lr2 * (update_dist_r - eps2 - opt.train.scheduler.delta2 * alpha2)
            # # alpha2 = torch.clamp(alpha2, min=0.0)
            # # print(alpha2)
            # alpha2 = alpha2.detach()
            # if alpha2.item() < 0.:
            #     alpha2 = torch.tensor(0).float().cuda()
            # print(alpha2)

        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss)
        losses.update(reduced_loss.item(), tot_rois)
        global_losses.update(reduced_loss.item(), tot_rois)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i + 1) % opt.print_freq == 0 and rank == 0:
            writer.add_scalar('train/loss', losses.avg, curr_step + 1)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], curr_step + 1)

            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': curr_step + 1,
                'loss': losses.avg,
                'lr': optimizer.param_groups[0]['lr']
            })

            logger.info('Epoch [{0}]\t'
                        'Iter [{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch,
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses))

    if rank == 0:
        writer.add_scalar('train/epoch_loss', global_losses.avg, epoch)
        writer.flush()

        epoch_logger.log({
            'epoch': epoch,
            'loss': global_losses.avg,
            'lr': optimizer.param_groups[0]['lr']
        })

        logger.info('-' * 100)
        logger.info(
            'Epoch [{}/{}]\t'
            'Loss {:.4f}'.format(
                epoch,
                opt.train.n_epochs,
                global_losses.avg))

        if epoch % opt.train.save_freq == 0:
            save_file_path = os.path.join(opt.result_path, 'ckpt_{}.pth.tar'.format(epoch))
            states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(states, save_file_path)
            logger.info('Checkpoint saved to {}'.format(save_file_path))

        logger.info('-' * 100)


def val_epoch(epoch, data_loader, model, criterion, act_func,
              opt, logger, epoch_logger, rank, world_size, writer):
    if rank == 0:
        logger.info('Evaluation at epoch {}'.format(epoch))

    model.eval()

    calc_loss = opt.val.with_label
    out_file = open(os.path.join(opt.result_path, 'tmp', 'predict_rank%d.csv' % rank), 'w')

    batch_time = AverageMeter(opt.print_freq)
    data_time = AverageMeter(opt.print_freq)
    if calc_loss:
        global_losses = AverageMeter()

    end_time = time.time()
    for i, data in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        with torch.no_grad():
            ret = model(data, evaluate=True)
            num_rois = ret['num_rois']
            outputs = ret['outputs']
            targets = ret['targets']
            B_alpha = ret['B_alpha']
            B_beta = ret['B_beta']


        if num_rois == 0:
            end_time = time.time()
            continue

        if calc_loss:
            # loss = criterion(outputs, targets)
            loss = criterion(B_alpha, B_beta, targets)
            global_losses.update(loss.item(), num_rois)

        fnames, mid_times, bboxes = ret['filenames'], ret['mid_times'], ret['bboxes']
        # outputs = act_func(outputs).cpu().data

        outputs = act_func(B_alpha, B_beta).cpu().data


        idx_to_class = data_loader.dataset.idx_to_class
        for k in range(num_rois):
            prefix = "%s,%s,%.3f,%.3f,%.3f,%.3f" % (fnames[k], mid_times[k],
                                                    bboxes[k][0], bboxes[k][1],
                                                    bboxes[k][2], bboxes[k][3])
            for cls in range(outputs.shape[1]):
                score_str = '%.3f' % outputs[k][cls]
                out_file.write(prefix + ",%d,%s\n" % (idx_to_class[cls]['id'], score_str))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (i + 1) % opt.print_freq == 0 and rank == 0:
            logger.info('Epoch [{0}]\t'
                        'Iter [{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                epoch,
                i + 1,
                len(data_loader),
                batch_time=batch_time,
                data_time=data_time))

    if calc_loss:
        total_num = torch.Tensor([global_losses.count]).cuda()
        loss_sum = torch.Tensor([global_losses.avg * global_losses.count]).cuda()
        dist.all_reduce(total_num)
        dist.all_reduce(loss_sum)
        final_loss = loss_sum.item() / (total_num.item() + 1e-10)

    out_file.close()
    dist.barrier()

    if rank == 0:
        val_log = {'epoch': epoch}
        val_str = 'Epoch [{}]'.format(epoch)

        if calc_loss:
            writer.add_scalar('val/epoch_loss', final_loss, epoch)
            val_log['loss'] = final_loss
            val_str += '\tLoss {:.4f}'.format(final_loss)

        result_file = os.path.join(opt.result_path, 'predict_epoch%d.csv' % epoch)
        with open(result_file, 'w') as of:
            for r in range(world_size):
                with open(os.path.join(opt.result_path, 'tmp', 'predict_rank%d.csv' % r), 'r') as f:
                    of.writelines(f.readlines())

        if opt.val.get('eval_mAP', None) is not None:
            eval_mAP = opt.val.eval_mAP
            metrics = run_evaluation(
                open(eval_mAP.labelmap, 'r'),
                open(eval_mAP.groundtruth, 'r'),
                open(result_file, 'r'),
                open(eval_mAP.exclusions, 'r') if eval_mAP.get('exclusions', None) is not None else None,
                logger
            )

            mAP = metrics['PascalBoxes_Precision/mAP@0.5IOU']
            writer.add_scalar('val/mAP', mAP, epoch)
            val_log['mAP'] = mAP
            val_str += '\tmAP {:.6f}'.format(mAP)

        writer.flush()

        if epoch_logger is not None:
            epoch_logger.log(val_log)

            logger.info('-' * 100)
            logger.info(val_str)
            logger.info('-' * 100)

    dist.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch AVA Training and Evaluation')
    parser.add_argument('--config', type=str, default='configs/AVA/SLOWFAST_R50_ACAR_HR2O.yaml')
    # parser.add_argument('--config', type=str, default='configs/AVA/eval_SLOWFAST_R50_ACAR_HR2O.yaml')
    parser.add_argument('--nproc_per_node', type=int, default=4)
    parser.add_argument('--backend', type=str, default='gloo')
    parser.add_argument('--master_addr', type=str, default=socket.gethostbyname(socket.gethostname()))
    parser.add_argument('--master_port', type=int, default=31114)  # 1234 31114 31120
    parser.add_argument('--nnodes', type=int, default=None)
    parser.add_argument('--node_rank', type=int, default=None)
    args = parser.parse_args()

    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

    torch.multiprocessing.spawn(main, args=(args,), nprocs=args.nproc_per_node)


