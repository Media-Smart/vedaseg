import torch.nn as nn
import logging
from torchvision.models.resnet import model_urls

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from .registry import BACKBONES
from ...weight_init import init_weights
from ...utils.norm import build_norm_layer

logger = logging.getLogger()


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_cfg=dict(type='BN')):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = build_norm_layer(norm_cfg, planes, return_layer=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = build_norm_layer(norm_cfg, planes, return_layer=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_cfg=dict(type='BN')):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = build_norm_layer(norm_cfg, width, return_layer=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = build_norm_layer(norm_cfg, width, return_layer=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = build_norm_layer(norm_cfg, planes * self.expansion, return_layer=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


MODEL_CFGS = {
    'resnet101': {
        'block': Bottleneck,
        'layer': [3, 4, 23, 3],
        'weights_url': model_urls['resnet101'],
    },
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


class ResNetCls(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, multi_grid=None,
                 norm_cfg=dict(type='BN')):
        super(ResNetCls, self).__init__()
        self._norm_cfg = norm_cfg

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = build_norm_layer(self._norm_cfg, self.inplanes, return_layer=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], multi_grid=multi_grid)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, multi_grid=None):
        norm_cfg = self._norm_cfg
        downsample = None
        previous_dilation = self.dilation

        if multi_grid is None:
            multi_grid = [1 for _ in range(blocks)]
        else:
            assert len(multi_grid) == blocks

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                build_norm_layer(norm_cfg, planes * block.expansion, return_layer=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation*multi_grid[0], norm_cfg))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation*multi_grid[i],
                                norm_cfg=norm_cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


@BACKBONES.register_module
class ResNet(ResNetCls):
    """ResNetEncoder

    Args:
        pretrain(bool)
    """
    def __init__(self, arch, replace_stride_with_dilation=None, multi_grid=None, pretrain=True, norm_cfg=dict(type='BN')):
        cfg = MODEL_CFGS[arch]
        super().__init__(
            cfg['block'],
            cfg['layer'],
            replace_stride_with_dilation=replace_stride_with_dilation,
            multi_grid=multi_grid,
            norm_cfg=norm_cfg)

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
