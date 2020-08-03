import torch
import numpy as np
import torch.nn.functional as F

from .inference_runner import InferenceRunner


class TestRunner(InferenceRunner):
    def __init__(self, test_cfg, inference_cfg, base_cfg=None):
        super().__init__(inference_cfg, base_cfg)

        self.test_dataloader = self._build_dataloader(test_cfg['data'])
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

                if self.tta:
                    output = self._tta_compute(image)
                else:
                    output = self.model(image)

                if isinstance(output, list):
                    output = [o.cpu().numpy() for o in output]
                else:
                    output = output.cpu().numpy()

                self.metric(output, mask.cpu().numpy())
                res = self.metric.accumulate()
                self.logger.info('Test, Iter {}, {}'.format(
                    idx + 1,
                    ', '.join(['{}: {}'.format(k, np.round(v, 4)) for k, v in
                               res.items()])))
        self.logger.info(', '.join(
            ['{}: {}'.format(k, np.round(v, 4)) for k, v in res.items()]))

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
            return probs  # t b c h w
        else:
            prob = torch.stack(probs, dim=0).softmax(dim=2).mean(dim=0)  # b c h w
            return prob
