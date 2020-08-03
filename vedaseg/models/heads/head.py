import logging

import torch.nn as nn

from .registry import HEADS
from ..utils import ConvModules, build_module
from ..weight_init import init_weights

logger = logging.getLogger()


@HEADS.register_module
class Head(nn.Module):
    """Head

    Args:
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels=None,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Relu', inplace=True),
                 num_convs=0,
                 upsample=None,
                 dropouts=None):
        super().__init__()

        if num_convs > 0:
            layers = [
                ConvModules(in_channels,
                            inter_channels,
                            3,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            num_convs=num_convs,
                            dropouts=dropouts),
                nn.Conv2d(inter_channels, out_channels, 1)
            ]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, 1)]
        if upsample:
            upsample_layer = build_module(upsample)
            layers.append(upsample_layer)

        self.block = nn.Sequential(*layers)
        logger.info('Head init weights')
        init_weights(self.modules())

    def forward(self, x):
        feat = self.block(x)
        return feat
