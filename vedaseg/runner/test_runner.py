import torch
import numpy as np
import torch.nn.functional as F

from .inference_runner import InferenceRunner
from ..utils import gather_tensor


class TestRunner(InferenceRunner):
    def __init__(self, test_cfg, inference_cfg, base_cfg=None):
        super().__init__(inference_cfg, base_cfg)

        self.test_dataloader = self._build_dataloader(test_cfg['data'])
        self.test_exclude_num = self.world_size - len(
            self.test_dataloader.dataset) % self.world_size
        self.tta = test_cfg.get('tta', False)

    def __call__(self):
        self.metric.reset()
        self.model.eval()

        res = {}

        self.logger.info('Start testing')
        with torch.no_grad():
            for idx, (image, mask) in enumerate(self.test_dataloader):
                if self.use_gpu:
                    image = image.cuda()
                    mask = mask.cuda()

                if self.tta:
                    output = self._tta_compute(image)
                else:
                    output = self.model(image)
                    output = self.compute(output)

                output = gather_tensor(output)
                mask = gather_tensor(mask)
                if idx + 1 == len(
                        self.test_dataloader) and self.test_exclude_num > 0:
                    output = output[:-self.test_exclude_num]
                    mask = mask[:-self.test_exclude_num]

                self.metric(output.cpu().numpy(), mask.cpu().numpy())
                res = self.metric.accumulate()
                self.logger.info('Test, Iter {}, {}'.format(
                    idx + 1,
                    ', '.join(['{}: {}'.format(k, np.round(v, 4)) for k, v in
                               res.items()])))
        self.logger.info('Test Result: {}'.format(', '.join(
            ['{}: {}'.format(k, np.round(v, 4)) for k, v in res.items()])))

        return res

    def _tta_compute(self, image):
        b, c, h, w = image.size()
        probs = []
        for scale, bias in zip(self.tta['scales'], self.tta['biases']):
            new_h, new_w = int(h * scale + bias), int(w * scale + bias)
            new_img = F.interpolate(image, size=(new_h, new_w),
                                    mode='bilinear', align_corners=True)
            output = self.model(new_img)
            probs.append(output)

            if self.tta['flip']:
                flip_img = new_img.flip(3)
                flip_output = self.model(flip_img)
                prob = flip_output.flip(3)
                probs.append(prob)

        if self.multi_label:
            prob = torch.stack(probs, dim=0).sigmoid().mean(dim=0)
            prob = torch.where(prob >= 0.5,
                               torch.full_like(prob, 1),
                               torch.full_like(prob, 0)).long()  # b c h w
        else:
            prob = torch.stack(probs, dim=0).softmax(dim=2).mean(dim=0)
            _, prob = torch.max(prob, dim=1)  # b h w
        return prob
