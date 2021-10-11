import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, skip=False):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv =  nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256+320 if skip else 256, num_classes, kernel_size=1, stride=1)
        )
        self.skip = skip

        self._init_weight()


    def forward(self, x, low_level_feat, x_encoder):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        for module in self.last_conv[:-1]:
            x = module(x)
        if self.skip:
            x_encoder = F.interpolate(x_encoder, size=x.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, x_encoder), dim=1)
        self.last_layer = x
        x = self.last_conv[-1](x)
        return x


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # self.class_projection.weight.data *= self.last_proj_factor


class Decoder0(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, last_proj_factor=1):
        super(Decoder0, self).__init__()

        print("Setting up decoder 0")

        self.class_projection = nn.Conv2d(320, num_classes, kernel_size=1, stride=1)

        self._init_weight()


    def forward(self, x, low_level_feat, x_encoder):
        self.last_layer = x
        x = self.class_projection(self.last_layer)
        return x


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, backbone, BatchNorm, v='3.1', *args, **kwargs):
    if v == '3.1':
        return Decoder(num_classes, backbone, BatchNorm, *args, **kwargs)
    if v == '3.2':
        return Decoder(num_classes, backbone, BatchNorm, *args, skip=True, **kwargs)
    return Decoder0(num_classes, backbone, BatchNorm, *args, **kwargs)
