import torch
import alphaexpansion
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys, warnings
from datetime import datetime


class AlphaExpansion(nn.Module):
    def __init__(self, max_iter, potts_weight, ce_weight=1, restrict=False, scale=1):
        super(AlphaExpansion, self).__init__()
        self.max_iter, self.potts_weight, self.ce_weight = max_iter, potts_weight, ce_weight
        self.restrict = restrict
        self.scale_factor = scale
        print("AlphaExpansion module is set up")


    def forward(self, unary, images, ROI, seeds, x0=None, **_):
        is_cuda = unary.is_cuda
        if ROI.ndim < 4:
            ROI = ROI[:, None, :, :]
        if seeds.ndim < 4:
            seeds = seeds[:, None, :, :]
        assert self.scale_factor <= 1
        if self.scale_factor < 1:
            warnings.warn("AlphaExpansion: scale_factor is not 1, the interpolated result may suffer from rounding errors")
            orig_size = images.shape[-2:]
            kwargs = {"scale_factor": self.scale_factor, "recompute_scale_factor": False}
            images = F.interpolate(images, **kwargs)
            unary = F.interpolate(unary, mode='bilinear', align_corners=False, **kwargs)
            ROI = F.interpolate(ROI, **kwargs)
            seeds = F.interpolate(seeds.float(), **kwargs).byte()
            if x0 is not None:
                x0 = F.interpolate(x0.float(), **kwargs).byte()
                x0[seeds != 255] = seeds[seeds != 255]

        unary = unary * self.ce_weight

        if self.restrict:
            N, C = unary.shape[:2]
            if N > 1:
                present = seeds.clone().long().reshape(N, -1)
                present[present == 255] = 0
                restricted = torch.zeros([N, C]).scatter(1, present, 1) == 0
                unary[restricted] = self.potts_weight * 9
            else:
                present = seeds.long().unique()
                if (present != 255).byte().sum() <= 1:
                    present = torch.arange(C+1, dtype=torch.int64)
                    present[-1] = 255
                to_new_label = torch.cumsum(
                    torch.zeros(256).scatter(0, present, 1),
                    dim=0
                ) - 1
                unlabeled = to_new_label[255].item()
                to_new_label[255] = 255
                seeds = torch.index_select(to_new_label, 0, seeds.reshape(-1).long()).reshape(seeds.shape)
                if x0 is not None:
                    x0 = torch.index_select(to_new_label, 0, x0.reshape(-1).long()).reshape(x0.shape)
                if present[-1] == 255:
                    unary = unary[:, present[:-1], ...]
                else:
                    unary = unary[:, present, ...]

        out = np.zeros(seeds.shape, np.float32)
        unary_energy = np.zeros(seeds.shape[:1], np.float32)
        smooth_energy = np.zeros(seeds.shape[:1], np.float32)
        images, ROI, seeds, unary = [x.detach().cpu().numpy() for x in [images, ROI, seeds, unary]]
        if x0 is None:
            x0 = np.zeros(seeds.shape, np.float32)
            # x0 = np.argmin(unary, 1)[:,None,:,:].astype(np.float32)
        else:
            x0 = x0.numpy()
        alphaexpansion.run_expansion(
            images, x0, ROI, seeds, unary,
            self.max_iter, self.potts_weight, out, unary_energy, smooth_energy)
        out[ROI == 0] = 255
        result = torch.tensor(out)

        if self.restrict:
            if N > 1:
                present2 = result.reshape(N, -1)
                present2[present == 255] = 0
                restricted2 = torch.zeros([N, C]).scatter(1, present2.long(), 1) == 0
                if (restricted & ~restricted2).any():
                    print ("Failed to respect the label restriction")
            else:
                result[result == 255] = unlabeled
                result = present[result.reshape(-1).long()].reshape(result.shape)

        if is_cuda:
            result = result.cuda()
        if self.scale_factor < 1:
            result = F.interpolate(result.float(), size=orig_size)

        return result.byte(), torch.tensor(unary_energy), torch.tensor(smooth_energy)
