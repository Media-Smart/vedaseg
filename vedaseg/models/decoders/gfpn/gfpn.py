import logging
import torch.nn as nn

from ...weight_init import init_weights
from ..builder import build_brick, build_bricks
from ..registry import DECODERS

logger = logging.getLogger()


@DECODERS.register_module
class GFPN(nn.Module):
    """GFPN
    A general framework for FPN-alike structures.
    """

    def __init__(self, neck, fusion=None):
        """
        Args:
            neck: cfg that describes the structure of GFPN

            fusion: cfg that describes the fusion behaviour of GFPN
        """
        super().__init__()
        self.neck = build_bricks(neck)
        if fusion:
            self.fusion = build_brick(fusion)
        else:
            self.fusion = None
        logger.info('GFPN init weights')
        init_weights(self.modules())

    def forward(self, bottom_up):
        """
        Args:
            bottom_up: dict of features from backbone
        """
        x = None
        feats = {**bottom_up}
        for ii, layer in enumerate(self.neck):
            if layer.to_layer in feats:
                raise KeyError(f'Layer name {layer.to_layer} already in use. '
                               f'Used names are: {list(feats.keys())}.')

            vertical_sources = layer.from_layers.get('vertical')
            lateral_sources = layer.from_layers.get('lateral')
            lateral_in, vertical_in = [], []

            if lateral_sources is not None and len(lateral_sources) > 0:
                for l_source in lateral_sources:
                    lateral_in.append(feats[l_source])

            if vertical_sources is not None and len(vertical_sources) > 0:
                for v_source in vertical_sources:
                    vertical_in.append(feats[v_source])

            x = layer(vertical_in, lateral_in)
            feats[layer.to_layer] = x
        if self.fusion:
            x = self.fusion(feats)
        return x
