import torch
import torch.nn as nn
from collections import defaultdict
from copy import deepcopy
from math import log2

from ..utils import ConvModule, build_module
from .registry import BRICKS

FUSION_METHODS = ('concat', 'add')


@BRICKS.register_module
class JunctionBlock(nn.Module):
    """JunctionBlock
    A basic building block in GFPN who process multiple input features and then
    fuse them into one output feature.

    Note:
        The reasons we separate incoming sources into 'vertials' & 'laterals':
            1. We don't want the structure to be too general, we'd like to
                keep the idea of 'lateral connection' / 'vertical
                connection' in the FPN structure.
            2. It's a little bit easier to navigate though the config file
                since connections are now categorized into two types.
    """

    def __init__(self, to_layer, verticals=None, laterals=None,
                 fusion_method=None):
        """
        Args:
            verticals: List of cfgs that describe modules used to process each
                        incoming vertical connections.(only one top-down
                        connection in original FPN)
            laterals: List of cfgs that describe modules used to process each
                        incoming lateral connections.(only one lateral
                        connection in original FPN)
            to_layer: Name of the output fused features.

            fusion_method: Fusion method. currently support 'concat' &
                            'add'. When used in single-input-single-output
                            cases, fusion_method should be set to None.
        """
        super().__init__()
        self.from_layers = defaultdict(list)
        self.to_layer = to_layer
        if fusion_method not in FUSION_METHODS and fusion_method is not None:
            raise ValueError(f'Provided fusion method {fusion_method} not '
                             f'supported. Availables:{FUSION_METHODS}')
        self.fusion_method = fusion_method
        verticals_ = deepcopy(verticals)
        laterals_ = deepcopy(laterals)

        self.vertical_blocks = nn.ModuleList()
        if verticals_ is not None:
            for vertical in verticals_:
                self.from_layers['vertical'].append(vertical.pop('from_layer'))
                if len(vertical) > 0:
                    self.vertical_blocks.append(build_module(vertical))
                else:
                    self.vertical_blocks.append(nn.Identity())

        self.lateral_blocks = nn.ModuleList()
        if laterals_ is not None:
            for lateral in laterals_:
                self.from_layers['lateral'].append(lateral.pop('from_layer'))
                if len(lateral) > 0:
                    self.lateral_blocks.append(build_module(lateral))
                else:
                    self.lateral_blocks.append(nn.Identity())

    def forward(self, verticals=None, laterals=None):
        """
        Args:
            verticals: Features from vertical connections(e.g. top-down
                    connection in a general FPN).
            laterals: Features from lateral connections(e.g. lateral
                    connection in a general FPN).
        """
        feat, vertical_res, lateral_res = None, [], []
        if verticals is not None and len(verticals) > 0:
            assert len(verticals) == len(self.vertical_blocks)
            for v_in, v_block in zip(verticals, self.vertical_blocks):
                vertical_res.append(v_block(v_in))

        if laterals is not None and len(laterals) > 0:
            assert len(laterals) == len(self.lateral_blocks)
            for l_in, l_block in zip(laterals, self.lateral_blocks):
                lateral_res.append(l_block(l_in))

        res_feats = [*vertical_res, *lateral_res]
        if len(res_feats) == 0:
            raise ValueError('There is neither vertical feature nor lateral '
                             'feature provided')
        elif len(res_feats) == 1:
            assert self.fusion_method is None, ('No fusion method supportd '
                                                'for single input'
                                                '(No need to fuse).')
            feat = res_feats[0]
        else:
            assert self.fusion_method is not None, ('No fusion method '
                                                    'provided.')
            if self.fusion_method == 'concat':
                feat = torch.cat(res_feats, 1)
            elif self.fusion_method == 'add':
                feat = torch.stack(res_feats, 0).sum(0)

        return feat


@BRICKS.register_module
class FusionBlock(nn.Module):
    """FusionBlock

        Args:
    """

    def __init__(self,
                 method,
                 from_layers,
                 feat_strides,
                 in_channels_list,
                 out_channels_list,
                 upsample,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='Relu', inplace=True),
                 common_stride=4,
                 ):
        super().__init__()
        assert method in ('add', 'concat')
        self.method = method
        self.from_layers = from_layers

        assert len(in_channels_list) == len(out_channels_list)

        self.blocks = nn.ModuleList()
        for idx, _ in enumerate(from_layers):
            in_channels = in_channels_list[idx]
            out_channels = out_channels_list[idx]
            feat_stride = feat_strides[idx]
            ups_num = max(1, int(log2(feat_stride) - log2(common_stride)))
            head_ops = []
            for idx2 in range(ups_num):
                cur_in_channels = in_channels if idx2 == 0 else out_channels
                conv = ConvModule(
                    cur_in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                head_ops.append(conv)
                if int(feat_stride) != int(common_stride):
                    head_ops.append(build_module(upsample))
            self.blocks.append(nn.Sequential(*head_ops))

    def forward(self, feats):
        outs = []
        for idx, key in enumerate(self.from_layers):
            block = self.blocks[idx]
            feat = feats[key]
            out = block(feat)
            outs.append(out)
        if self.method == 'add':
            res = torch.stack(outs, 0).sum(0)
        else:
            res = torch.cat(outs, 1)
        return res


@BRICKS.register_module
class CollectBlock(nn.Module):
    """FusionBlock

        Args:
    """

    def __init__(self, from_layer, to_layer=None):
        super().__init__()
        self.from_layer = from_layer
        self.to_layer = to_layer

    def forward(self, feats):
        if self.to_layer is None:
            return feats[self.from_layer]
        else:
            feats[self.to_layer] = feats[self.from_layer]
