# modify from mmcv and mmdetection

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class TLU(nn.Module):
    def __init__(self, num_features):
        super(TLU, self).__init__()

        self.num_features = num_features
        self.tau = Parameter(torch.Tensor(1, num_features, 1, 1),
                             requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def forward(self, x):
        return torch.max(x, self.tau)

    def extra_repr(self):
        return '{num_features}'.format(**self.__dict__)


act_cfg = {
    'Relu': ('relu', nn.ReLU),
    'Tlu': ('tlu', TLU),
}


def build_act_layer(cfg, num_features, postfix='', layer_only=False):
    """ Build activate layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify activate layer type.
            layer args: args needed to instantiate a activate layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into act abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created act layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in act_cfg:
        raise KeyError('Unrecognized activate type {}'.format(layer_type))
    else:
        abbr, act_layer = act_cfg[layer_type]
        if act_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    if layer_type != 'Tlu':
        layer = act_layer(**cfg_)
    else:
        layer = act_layer(num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    if layer_only:
        return layer
    else:
        return name, layer
