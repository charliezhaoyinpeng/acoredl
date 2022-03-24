import math

import torch
import torch.nn as nn
import torchvision

__all__ = ['acar']


class HR2O_NL(nn.Module):
    def __init__(self, hidden_dim=512, kernel_size=3, mlp_1x1=False):
        super(HR2O_NL, self).__init__()

        self.hidden_dim = hidden_dim

        padding = kernel_size // 2
        self.conv_q = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_k = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.conv_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.conv = nn.Conv2d(
            hidden_dim, hidden_dim,
            1 if mlp_1x1 else kernel_size,
            padding=0 if mlp_1x1 else padding,
            bias=False
        )
        self.norm = nn.GroupNorm(1, hidden_dim, affine=True)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        query = self.conv_q(x).unsqueeze(1)
        key = self.conv_k(x).unsqueeze(0)
        att = (query * key).sum(2) / (self.hidden_dim ** 0.5)
        att = nn.Softmax(dim=1)(att)
        value = self.conv_v(x)
        virt_feats = (att.unsqueeze(2) * value).sum(1)

        virt_feats = self.norm(virt_feats)
        virt_feats = self.lrelu(virt_feats)
        virt_feats = self.conv(virt_feats)
        virt_feats = self.dp(virt_feats)

        x = x + virt_feats
        return x


class ACARHead(nn.Module):
    def __init__(self, width, roi_spatial=7, num_classes=60, dropout=0., bias=False,
                 reduce_dim=1024, hidden_dim=512, downsample='max2x2', depth=2,
                 kernel_size=3, mlp_1x1=False):
        super(ACARHead, self).__init__()

        self.num_classes = num_classes
        self.roi_spatial = roi_spatial
        self.roi_maxpool = nn.MaxPool2d(roi_spatial)

        # actor-context feature encoder
        self.conv_reduce = nn.Conv2d(width, reduce_dim, 1, bias=False)
        self.gn = nn.GroupNorm(1, hidden_dim, affine=True)
        self.conv1 = nn.Conv2d(reduce_dim * 3, hidden_dim, 1, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.elu = nn.ELU()

        # down-sampling before HR2O
        assert downsample in ['none', 'max2x2']
        if downsample == 'none':
            self.downsample = nn.Identity()
        elif downsample == 'max2x2':
            self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # high-order relation reasoning operator (HR2O_NL)
        layers = []
        for _ in range(depth):
            layers.append(HR2O_NL(hidden_dim, kernel_size, mlp_1x1))
        self.hr2o = nn.Sequential(*layers)

        # classification
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(reduce_dim, hidden_dim, bias=False)
        # self.bn = nn.GroupNorm(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 2, num_classes, bias=bias)
        self.fc3 = nn.Linear(hidden_dim * 2, num_classes, bias=bias)

        if dropout > 0:
            self.dp = nn.Dropout(dropout)
        else:
            self.dp = None

    # data: features, rois, num_rois, roi_ids, sizes_before_padding
    # returns: outputs
    def forward(self, data):
        if not isinstance(data['features'], list):
            feats = [data['features']]
        else:
            feats = data['features']

        # person_ids = data['person_ids']
        # person_id_idx = []
        # for i in range(len(person_ids)):
        #     if person_ids[i] != -100:
        #         person_id_idx.append(i)
        # person_id_idx = torch.tensor(person_id_idx).cuda()

        # temporal average pooling
        h, w = feats[0].shape[3:]
        # requires all features have the same spatial dimensions
        feats = [nn.AdaptiveAvgPool3d((1, h, w))(f).view(-1, f.shape[1], h, w) for f in feats]
        feats = torch.cat(feats, dim=1)

        feats = self.conv_reduce(feats)

        # rois features of both persons and objs
        # rois = data['rois']
        # rois[:, 1] = rois[:, 1] * w
        # rois[:, 2] = rois[:, 2] * h
        # rois[:, 3] = rois[:, 3] * w
        # rois[:, 4] = rois[:, 4] * h
        # rois = rois.detach()
        # roi_feats = torchvision.ops.roi_align(feats, rois, (self.roi_spatial, self.roi_spatial))
        # roi_feats = self.roi_maxpool(roi_feats).view(data['num_rois'], -1)

        # rois features of persons only
        rois_ps = data['rois_ps']
        rois_ps[:, 1] = rois_ps[:, 1] * w
        rois_ps[:, 2] = rois_ps[:, 2] * h
        rois_ps[:, 3] = rois_ps[:, 3] * w
        rois_ps[:, 4] = rois_ps[:, 4] * h
        rois_ps = rois_ps.detach()
        roi_ps_feats = torchvision.ops.roi_align(feats, rois_ps, (self.roi_spatial, self.roi_spatial))
        roi_ps_feats = self.roi_maxpool(roi_ps_feats).view(data['num_rois_ps'], -1)
        num_rois_ps = data['num_rois_ps']

        # rois features of objs only
        rois_obj = data['rois_obj']
        num_rois_obj = data['num_rois_obj']
        if num_rois_obj != 0:
            rois_obj[:, 1] = rois_obj[:, 1] * w
            rois_obj[:, 2] = rois_obj[:, 2] * h
            rois_obj[:, 3] = rois_obj[:, 3] * w
            rois_obj[:, 4] = rois_obj[:, 4] * h
            rois_obj = rois_obj.detach()
            roi_obj_feats = torchvision.ops.roi_align(feats, rois_obj, (self.roi_spatial, self.roi_spatial))
            roi_obj_feats = self.roi_maxpool(roi_obj_feats).view(data['num_rois_obj'], -1)
            lcm = int(num_rois_ps * num_rois_obj / math.gcd(num_rois_ps, num_rois_obj))
            obj_feats = roi_obj_feats
        else:
            roi_obj_feats = torch.zeros_like(roi_ps_feats).cuda()
            lcm = num_rois_ps
            obj_feats = None

        roi_ids = data['roi_ids']
        roi_ps_ids = data['roi_ps_ids']
        roi_obj_ids = data['roi_obj_ids']
        sizes_before_padding = data['sizes_before_padding']
        high_order_feats = []

        # print(roi_ids, roi_ps_ids, num_rois_obj, "=================")
        # print(roi_ps_feats.shape)
        # try:
        #     print(roi_obj_feats.shape)
        # except:
        #     print(None)

        for idx in range(feats.shape[0]):  # iterate over mini-batch
            n_ps_rois = data['num_rois_ps']
            if n_ps_rois == 0:
                continue

            eff_h, eff_w = math.ceil(h * sizes_before_padding[idx][1]), math.ceil(w * sizes_before_padding[idx][0])
            bg_feats = feats[idx][:, :eff_h, :eff_w]
            repeat_bg_feats = bg_feats.unsqueeze(0).repeat((n_ps_rois, 1, 1, 1))
            reduce = nn.AdaptiveAvgPool2d((lcm, self.num_classes))
            reduce_bg_feats = reduce(bg_feats).mean(0)

            # print("!!!!!!!!!", bg_feats.shape, repeat_bg_feats.shape, reduce_bg_feats.shape)

            # actor_feats = roi_ps_feats[roi_ids[idx]:roi_ids[idx + 1]]
            tiled_ps_feats = roi_ps_feats.unsqueeze(2).unsqueeze(2).expand_as(repeat_bg_feats)
            a, b, c, d = repeat_bg_feats.shape
            tiled_obj_feats = roi_obj_feats.unsqueeze(2).unsqueeze(2).expand(-1, -1, c, d)
            # print("+++++++++", tiled_ps_feats.shape, tiled_obj_feats.shape)

            # tiled_ps_feats = ps_feats.unsqueeze(2).unsqueeze(2).expand_as(bg_feats)

            # if len(rois_obj) != 0:
            #     obj_feats = roi_obj_feats
            #     # tiled_obj_feats = obj_feats.unsqueeze(2).unsqueeze(2).expand_as(bg_feats)
            # else:
            #     obj_feats = None

            interact_feats = torch.cat([repeat_bg_feats, tiled_ps_feats], dim=1)
            if num_rois_obj != 0:
                interact_feats = interact_feats.repeat(int(lcm / num_rois_ps), 1, 1, 1)
                tiled_obj_feats = tiled_obj_feats.repeat(int(lcm / num_rois_obj), 1, 1, 1)

            # print("+++++++++", interact_feats.shape, tiled_obj_feats.shape)
            interact_feats = torch.cat([interact_feats, tiled_obj_feats], dim=1)

            # print("~~~~~~~~~~~~~~~~", interact_feats.shape)

            interact_feats = self.conv1(interact_feats)
            # interact_feats = self.gn(interact_feats)
            interact_feats = self.lrelu(interact_feats)
            # interact_feats = self.gn(interact_feats)
            # print("@@@@@@@@@@@@@@@@", interact_feats.shape)

            interact_feats = self.conv2(interact_feats)
            interact_feats = self.gn(interact_feats)
            interact_feats = self.lrelu(interact_feats)

            interact_feats = self.downsample(interact_feats)
            # print("1111111111111", interact_feats.shape)
            interact_feats = self.hr2o(interact_feats)
            interact_feats = self.gap(interact_feats)

            high_order_feats.append(interact_feats)

        if len(high_order_feats) == 0:
            print("this high_order_feats is empty !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        high_order_feats = torch.cat(high_order_feats, dim=0).view(lcm, -1)
        # print("222222222", high_order_feats.shape)
        # high_order_feats = torch.index_select(high_order_feats, 0, person_id_idx)

        outputs = self.fc1(roi_ps_feats)
        # outputs = self.bn(outputs)
        outputs = self.lrelu(outputs)

        # print("333333333333", outputs.shape)
        outputs = outputs.repeat(int(lcm / num_rois_ps), 1)
        outputs = torch.cat([outputs, high_order_feats], dim=1)

        if self.dp is not None:
            outputs = self.dp(outputs)

        temp_outputs = outputs

        outputs = self.fc2(temp_outputs)
        B_alpha = self.elu(outputs) + 2
        B_beta = self.elu(self.fc3(temp_outputs)) + 2

        return {'outputs': outputs, 'bg_feats': bg_feats, 'ps_feats': roi_ps_feats, 'obj_feats': obj_feats,
                'reduce_bg_feats': reduce_bg_feats, 'B_alpha': B_alpha, 'B_beta': B_beta}


def acar(**kwargs):
    model = ACARHead(**kwargs)
    return model
