# modify from https://github.com/pytorch/vision/tree/master/torchvision/models/segmentation/deeplabv3.py

import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import ENHANCE_MODULES
from ...utils.act import build_act_layer
from ...utils.norm import build_norm_layer
from ...weight_init import init_weights

logger = logging.getLogger()


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, norm_layer,
                 act_layer):
        modules = [
            nn.Conv2d(in_channels,
                      out_channels,
                      3,
                      padding=dilation,
                      dilation=dilation,
                      bias=False),
            norm_layer(out_channels),
            act_layer(out_channels)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_layer, act_layer,
                 mode='bilinear', align_corners=True):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels), act_layer(out_channels))
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        y = super(ASPPPooling, self).forward(x)
        return F.interpolate(y,
                             size=(int(x.size(2)), int(x.size(3))),
                             mode=self.mode,
                             align_corners=self.align_corners)


@ENHANCE_MODULES.register_module
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, from_layer,
                 to_layer, mode='bilinear', align_corners=True, dropout=None,
                 norm_cfg=None, act_cfg=None):
        super(ASPP, self).__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer

        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        norm_layer = partial(build_norm_layer, norm_cfg, layer_only=True)

        if act_cfg is None:
            act_cfg = dict(type='Relu', inplace=True)
        act_layer = partial(build_act_layer, act_cfg, layer_only=True)

        modules = [nn.Sequential(nn.Conv2d(in_channels,
                                           out_channels,
                                           1,
                                           bias=False),
                                 norm_layer(out_channels),
                                 act_layer(out_channels))]

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(
            ASPPConv(in_channels, out_channels, rate1, norm_layer, act_layer))
        modules.append(
            ASPPConv(in_channels, out_channels, rate2, norm_layer, act_layer))
        modules.append(
            ASPPConv(in_channels, out_channels, rate3, norm_layer, act_layer))
        modules.append(
            ASPPPooling(in_channels, out_channels, norm_layer, act_layer,
                        mode=mode, align_corners=align_corners))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels), act_layer(out_channels))
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
