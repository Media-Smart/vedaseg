# modify from https://github.com/pytorch/vision/tree/master/torchvision/models/segmentation/deeplabv3.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .registry import ENHANCE_MODULES
from ...weight_init import init_weights
from ...utils.norm import build_norm_layer

logger = logging.getLogger()


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, norm_cfg=dict(type='BN')):
        modules = [
            nn.Conv2d(in_channels,
                      out_channels,
                      3,
                      padding=dilation,
                      dilation=dilation,
                      bias=False),
            build_norm_layer(norm_cfg, out_channels, return_layer=True),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_cfg=dict(type='BN')):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            build_norm_layer(norm_cfg, out_channels, return_layer=True), nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


@ENHANCE_MODULES.register_module
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, from_layer,
                 to_layer, dropout=None, norm_cfg=dict(type='BN')):
        super(ASPP, self).__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          build_norm_layer(norm_cfg, out_channels, return_layer=True), nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1, norm_cfg))
        modules.append(ASPPConv(in_channels, out_channels, rate2, norm_cfg))
        modules.append(ASPPConv(in_channels, out_channels, rate3, norm_cfg))
        modules.append(ASPPPooling(in_channels, out_channels, norm_cfg))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            build_norm_layer(norm_cfg, out_channels, return_layer=True), nn.ReLU(inplace=True))
        self.with_dropout = dropout is not None
        if self.with_dropout:
            self.dropout = nn.Dropout(dropout)

        logger.info('ASPP init weights')
        init_weights(self.modules())

    def forward(self, feats):
        feats_ = feats.copy()
        x = feats_[self.from_layer]
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.project(res)
        if self.with_dropout:
            res = self.dropout(res)
        feats_[self.to_layer] = res
        return feats_
