import torch

from ..models import build_model
from ..utils import load_checkpoint
from .base import Common


class InferenceRunner(Common):
    def __init__(self, inference_cfg, base_cfg=None):
        inference_cfg = inference_cfg.copy()
        base_cfg = {} if base_cfg is None else base_cfg.copy()

        base_cfg['gpu_id'] = inference_cfg.pop('gpu_id')
        super().__init__(base_cfg)

        self.multi_label = inference_cfg.get('multi_label', False)

        # build inference transform
        self.transform = self._build_transform(inference_cfg['transforms'])

        # build model
        self.model = self._build_model(inference_cfg['model'])
        self.model.eval()

    def load_checkpoint(self, filename, map_location='default', strict=True):
        self.logger.info('Load checkpoint from {}'.format(filename))

        if map_location == 'default':
            if self.use_gpu:
                device_id = torch.cuda.current_device()
                map_location = lambda storage, loc: storage.cuda(device_id)
            else:
                map_location = 'cpu'

        return load_checkpoint(self.model, filename, map_location, strict)

    def _build_model(self, cfg):
        self.logger.info('Build model')

        model = build_model(cfg)
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
            model.cuda()

        return model

    def _compute(self, output):
        if self.multi_label:
            output = output.sigmoid()
            output = torch.where(output >= 0.5,
                                 torch.full_like(output, 1),
                                 torch.full_like(output, 0))
            output = output.permute(0, 2, 3, 1)
        else:
            output = output.softmax(dim=1)
            _, output = torch.max(output, dim=1)
        return output

    def __call__(self, image, masks):
        with torch.no_grad():
            image = self.transform(image=image, masks=masks)['image']
            image = image.unsqueeze(0)

            if self.use_gpu:
                image = image.cuda()

                output = self.model(image)
                output = self._compute(output)

            output = output.squeeze().cpu().numpy()

        return output
