import logging
import torch.nn as nn
from torchvision.models.resnet import model_urls

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial

from ...utils.act import build_act_layer
from ...utils.norm import build_norm_layer
from ...weight_init import init_weights
from .registry import BACKBONES

logger = logging.getLogger()

model_urls.update(
    {
        "resnet18_v1c": "https://download.openmmlab.com/pretrain/third_party/"
                        "resnet18_v1c-b5776b93.pth",
        "resnet50_v1c": "https://download.openmmlab.com/pretrain/third_party/"
                        "resnet50_v1c-2cccc1ad.pth",
        "resnet101_v1c": "https://download.openmmlab.com/pretrain/third_party/"
                         "resnet101_v1c-e67eebb6.pth",
    }
)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_layer, act_layer, stride=1,
                 downsample=None, groups=1,
                 base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = act_layer(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.relu2 = act_layer(planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, norm_layer, act_layer, stride=1,
                 downsample=None, groups=1,
                 base_width=64, dilation=1, ):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu1 = act_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = act_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu3 = act_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


MODEL_CFGS = {
    'resnet101': {
        'block': Bottleneck,
        'layer': [3, 4, 23, 3],
    },
    'resnet50': {
        'block': Bottleneck,
        'layer': [3, 4, 6, 3],
    },
    'resnet18': {
        'block': BasicBlock,
        'layer': [2, 2, 2, 2],
    }
}


class ResNetCls(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 zero_init_residual=False,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, multi_grid=None,
                 norm_cfg=None, act_cfg=None, deep_stem=False):
        super(ResNetCls, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        self._norm_layer = partial(build_norm_layer, norm_cfg, layer_only=True)

        if act_cfg is None:
            act_cfg = dict(type='Relu', inplace=True)
        self._act_layer = partial(build_act_layer, act_cfg, layer_only=True)

        self.deep_stem = deep_stem
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None or a 3-element '
                'tuple, got {}'.format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group
        self._make_stem_layer()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       multi_grid=multi_grid)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the
        # residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                    multi_grid=None):
        norm_layer = self._norm_layer
        act_layer = self._act_layer
        downsample = None

        if multi_grid is None:
            multi_grid = [1 for _ in range(blocks)]
        else:
            assert len(multi_grid) == blocks

        if dilate:
            self.dilation *= stride
            stride = 1

        previous_dilation = self.dilation

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, norm_layer, act_layer,
                        stride, downsample, self.groups, self.base_width,
                        previous_dilation * multi_grid[0])]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer,
                                act_layer=act_layer,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation * multi_grid[i], ))

        return nn.Sequential(*layers)

    def _make_stem_layer(self):
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(3, self.inplanes // 2,
                          kernel_size=3, stride=2, padding=1, bias=False),
                self._norm_layer(self.inplanes // 2),
                self._act_layer(self.inplanes // 2),

                nn.Conv2d(self.inplanes // 2, self.inplanes // 2,
                          kernel_size=3, stride=1, padding=1, bias=False),
                self._norm_layer(self.inplanes // 2),
                self._act_layer(self.inplanes // 2),

                nn.Conv2d(self.inplanes // 2, self.inplanes,
                          kernel_size=3, stride=1, padding=1, bias=False),
                self._norm_layer(self.inplanes),
                self._act_layer(self.inplanes),
            )
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                                   padding=3, bias=False)
            self.bn1 = self._norm_layer(self.inplanes)
            self.relu1 = self._act_layer(self.inplanes)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
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

    def __init__(self, arch, replace_stride_with_dilation=None,
                 multi_grid=None, pretrain=True, norm_cfg=None, act_cfg=None):
        cfg = MODEL_CFGS[arch[:-4] if arch.endswith('_v1c') else arch]

        super().__init__(
            cfg['block'],
            cfg['layer'],
            replace_stride_with_dilation=replace_stride_with_dilation,
            deep_stem=arch.endswith('_v1c'),
            multi_grid=multi_grid,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if pretrain:
            logger.info('ResNet init weights from pretreain')
            if arch not in model_urls:
                raise KeyError('No model url exist for {}'.format(arch))
            state_dict = load_state_dict_from_url(model_urls[arch])
            if 'state_dict' in state_dict:
                # handle state_dict format from mmseg
                state_dict = state_dict['state_dict']
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.info('ResNet init weights')
            init_weights(self.modules())

        del self.fc, self.avgpool

    def forward(self, x):
        feats = {}
        if self.deep_stem:
            x0 = self.stem(x)
        else:
            x0 = self.conv1(x)
            x0 = self.bn1(x0)
            x0 = self.relu1(x0)
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

        return feats
