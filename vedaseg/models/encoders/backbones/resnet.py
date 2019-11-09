import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import ResNet as ResNetCls
from torchvision.models.resnet import model_urls

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from .registry import BACKBONES
from ...weight_init import init_weights

logger = logging.getLogger()

MODEL_CFGS = {
    'resnet50': {
        'block': Bottleneck,
        'layer': [3, 4, 6, 3],
        'weights_url': model_urls['resnet50'],
    },
    'resnet18': {
        'block': BasicBlock,
        'layer': [2, 2, 2, 2],
        'weights_url': model_urls['resnet18'],
    }
}


@BACKBONES.register_module
class ResNet(ResNetCls):
    """ResNetEncoder

    Args:
        pretrain(bool)
    """
    def __init__(self, arch, replace_stride_with_dilation=None, pretrain=True):
        cfg = MODEL_CFGS[arch]
        super().__init__(
            cfg['block'],
            cfg['layer'],
            replace_stride_with_dilation=replace_stride_with_dilation)

        if pretrain:
            logger.info('ResNet init weights from pretreain')
            state_dict = load_state_dict_from_url(cfg['weights_url'])
            self.load_state_dict(state_dict)
        else:
            logger.info('ResNet init weights')
            init_weights(self.modules())

        del self.fc, self.avgpool

    def forward(self, x):
        feats = {}

        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)  # 2
        feats['c1'] = x0

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)  # 4
        feats['c2'] = x1

        x2 = self.layer2(x1)  # 8
        feats['c3'] = x2
        x3 = self.layer3(x2)  # 16
        feats['c4'] = x3
        x4 = self.layer4(x3)  # 32
        feats['c5'] = x4

        #for k, v in feats.items():
        #    print(k, v.shape)

        return feats
