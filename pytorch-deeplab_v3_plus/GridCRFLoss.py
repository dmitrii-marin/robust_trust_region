import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import math
from dataloaders.custom_transforms import denormalizeimage
from itertools import repeat


class BilinearPottsRelaxation(object):
    @staticmethod
    def comute(a, b):
        return a * (1 - b)


class TVPottsRelaxation(object):
    @staticmethod
    def comute(a, b):
        return torch.abs(a - b)

class SquaredPottsRelaxation(object):
    @staticmethod
    def comute(a, b):
        return (a - b) ** 2


class GridPottsLoss(nn.Module):
    def __init__(self, weight, scale_factor, relaxation=BilinearPottsRelaxation, neighbourhood=8):
        super(GridPottsLoss, self).__init__()
        self.weight = weight
        self.scale_factor = scale_factor
        self.rel = relaxation
        SQRT2 = math.sqrt(2)
        if neighbourhood == 4:
            self.neighbourhood = [(0, 1, 1), (1, 0, 1)]
        elif neighbourhood == 8:
            self.neighbourhood = [(0, 1, 1), (1, 0, 1), (1, 1, SQRT2), (-1, 1, SQRT2)]
        else:
            raise Exception("Unknown neighbourhood: %d" % neighbourhood)


    def forward(self, images, segmentations, ROIs):
        if self.weight == 0:
            self.max_weight = torch.tensor(1, device=segmentations.device)
            result = torch.tensor(0, dtype=segmentations.dtype, device=segmentations.device)
            return result


        def get_diff(val, dx, dy, op=torch.sub):
            shape = val.shape
            h, w = shape[-2:]
            return op(val[..., max(0,-dx):min(h,h-dx), max(0,-dy):min(w,w-dy)],
                      val[..., max(0,dx):min(h,h+dx), max(0,dy):min(w,w+dy)])
            # return op(val[..., :h-dx, :w-dy], val[..., dx:, dy:])

        # scale imag by scale_factor
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor)
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor)

        use_cuda = segmentations.is_cuda

        sigma2 = 0
        count = 0
        for dx, dy, _ in self.neighbourhood:
            new_rois = get_diff(scaled_ROIs, dx, dy, torch.min)
            rgb_diff = get_diff(scaled_images, dx, dy) ** 2 * new_rois
            sigma2 += torch.sum(rgb_diff, (1,2,3), keepdim=True)
            count += torch.sum(new_rois, (1,2,3), keepdim=True)
        sigma2 = sigma2 / count
        sigma2[count == 0] = 1
        sigma2[sigma2 == 0] = 1
        sigma2 *= 2

        count = 0
        loss = 0
        max_weight = None
        for dx, dy, f in self.neighbourhood:
            new_rois = get_diff(scaled_ROIs, dx, dy, torch.min)
            rgb_diff = torch.sum(get_diff(scaled_images, dx, dy) ** 2, 1, keepdim=True)
            rgb_weight = new_rois * torch.exp(-rgb_diff / sigma2) / f
            if use_cuda:
                rgb_weight = rgb_weight.cuda()
            loc_max_weight, _ = torch.max(rgb_weight[:,0,:,:], 1)
            loc_max_weight, _ = torch.max(loc_max_weight, 1)
            max_weight = loc_max_weight if max_weight is None else torch.max(loc_max_weight, max_weight)
            pixel_loss = get_diff(scaled_segs, dx, dy, self.rel.comute) * rgb_weight
            count += torch.sum(new_rois, (1,2,3), keepdim=True)
            loss += torch.sum(pixel_loss, (1,2,3), keepdim=True)

        self.max_weight = max_weight * self.weight

        count[count == 0] = 1
        if use_cuda:
            count = count.cuda()
        loss /= count

        assert not torch.isnan(loss).any()

        return torch.mean(loss) * self.weight


    def extra_repr(self):
        return 'weight={}, scale_factor={}, neighborhood={}, relaxation={}'.format(
            self.weight, self.scale_factor, len(self.neighbourhood) * 2, self.rel
        )
