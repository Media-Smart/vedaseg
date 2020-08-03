# modify from mmcv and mmdetection

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        super(FRN, self).__init__()

        self.num_features = num_features
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1),
                               requires_grad=True)
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1),
                              requires_grad=True)

        self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x):
        nu2 = torch.mean(x.pow(2), dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps.abs())
        x = self.gamma * x + self.beta

        return x

    def extra_repr(self):
        return '{num_features}, eps={eps}'.format(**self.__dict__)


norm_cfg = {
    'FRN': ('frn', FRN),
    # format: layer_type: (abbreviation, module)
    'BN': ('bn', nn.BatchNorm2d),
    'SyncBN': ('bn', nn.SyncBatchNorm),
    'GN': ('gn', nn.GroupNorm),
    # and potentially 'SN'
}


def build_norm_layer(cfg, num_features, postfix='', layer_only=False):
    """ Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)  # noqa
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    if layer_only:
        return layer
    return name, layer
