import torch
import torch.nn as nn
import math
from .utils import bbox_jitter, get_bbox_after_aug

__all__ = ['basic']


class BasicNeck(nn.Module):
    def __init__(self, aug_threshold=0., bbox_jitter=None, num_classes=60, multi_class=True):
        super(BasicNeck, self).__init__()

        # threshold on preserved ratio of bboxes after cropping augmentation
        self.aug_threshold = aug_threshold
        # config for bbox jittering
        self.bbox_jitter = bbox_jitter

        self.num_classes = num_classes
        self.multi_class = multi_class

    # data: aug_info, labels, filenames, mid_times
    # returns: num_rois, rois, roi_ids, targets, sizes_before_padding, filenames, mid_times, bboxes, bbox_ids
    def forward(self, data):
        rois, roi_ids, targets, sizes_before_padding, filenames, mid_times, rois_ps, roi_ps_ids, rois_obj, roi_obj_ids = [], [
            0], [], [], [], [], [], [], [], []
        bboxes, bbox_ids = [], []  # used for multi-crop fusion

        person_ids = []

        cur_bbox_id = -1  # record current bbox no.
        for idx in range(len(data['aug_info'])):
            aug_info = data['aug_info'][idx]
            pad_ratio = aug_info['pad_ratio']
            sizes_before_padding.append([1. / pad_ratio[0], 1. / pad_ratio[1]])

            for label in data['labels'][idx]:
                cur_bbox_id += 1
                if self.training and self.bbox_jitter is not None:
                    bbox_list = bbox_jitter(label['bounding_box'],
                                            self.bbox_jitter.get('num', 1),
                                            self.bbox_jitter.scale)
                else:
                    # no bbox jittering during evaluation
                    bbox_list = [label['bounding_box']]

                for b in bbox_list:
                    bbox = get_bbox_after_aug(aug_info, b, self.aug_threshold)
                    if bbox is None:
                        continue
                    try:
                        this_person_id = label['person_id'][0]
                    except:
                        this_person_id = -100
                    if this_person_id != -100:  # this bbox indicates a person
                        rois_ps.append([idx] + bbox)
                    else:  # this bbox indicates an object
                        rois_obj.append([idx] + bbox)

                    rois.append([idx] + bbox)
                    filenames.append(data['filenames'][idx])
                    mid_times.append(data['mid_times'][idx])
                    bboxes.append(label['bounding_box'])
                    bbox_ids.append(cur_bbox_id)

                    try:
                        person_ids.append(label['person_id'][0])
                    except:
                        person_ids.append(-100)

                    if this_person_id != -100:
                        if self.multi_class:
                            ret = torch.zeros(self.num_classes)
                            ret.put_(torch.LongTensor(label['label']),
                                     torch.ones(len(label['label'])))
                        else:
                            ret = torch.LongTensor(label['label'])
                        targets.append(ret)

            roi_ps_ids.append(len(rois_ps))
            roi_obj_ids.append(len(rois_obj))
            roi_ids.append(len(rois))

        num_rois = len(rois)
        num_rois_ps = len(rois_ps)
        num_rois_obj = len(rois_obj)

        if num_rois_ps == 0:  # if there is no person
            return {'num_rois': 0, 'rois': None, 'roi_ids': roi_ids, 'targets': None,
                    'sizes_before_padding': sizes_before_padding,
                    'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes, 'bbox_ids': bbox_ids,
                    'person_ids': person_ids, 'rois_ps': None, 'num_rois_ps': 0,
                    'roi_ps_ids': roi_ps_ids, 'rois_obj': None, 'num_rois_obj': 0,
                    'roi_obj_ids': roi_obj_ids}

        rois = torch.FloatTensor(rois).cuda()
        rois_ps = torch.FloatTensor(rois_ps).cuda()
        rois_obj = torch.FloatTensor(rois_obj).cuda()
        targets = torch.stack(targets, dim=0).cuda()
        if num_rois_obj != 0:
            lcm = int(num_rois_ps * num_rois_obj / math.gcd(num_rois_ps, num_rois_obj))
            targets = targets.repeat(int(lcm / num_rois_ps), 1)
            # print("44444444444", targets.shape)

        return {'num_rois': num_rois, 'rois': rois, 'roi_ids': roi_ids, 'targets': targets,
                'sizes_before_padding': sizes_before_padding,
                'filenames': filenames, 'mid_times': mid_times, 'bboxes': bboxes, 'bbox_ids': bbox_ids,
                'person_ids': person_ids, 'rois_ps': rois_ps, 'num_rois_ps': num_rois_ps,
                'roi_ps_ids': roi_ps_ids, 'rois_obj': rois_obj, 'num_rois_obj': num_rois_obj,
                'roi_obj_ids': roi_obj_ids}


def basic(**kwargs):
    model = BasicNeck(**kwargs)
    return model
