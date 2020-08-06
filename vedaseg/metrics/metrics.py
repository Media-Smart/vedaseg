import numpy as np

from .base import BaseMetric
from .registry import METRICS


class Compose:
    def __init__(self, metrics):
        self.metrics = metrics

    def reset(self):
        for m in self.metrics:
            m.reset()

    def accumulate(self):
        res = dict()
        for m in self.metrics:
            mtc = m.accumulate()
            res.update(mtc)
        return res

    def __call__(self, pred, target):
        for m in self.metrics:
            m(pred, target)


class ConfusionMatrix(BaseMetric):
    """
    Calculate confusion matrix for segmentation
    Args:
        num_classes (int): number of classes.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__()

    def reset(self):
        self.cfsmtx = np.zeros((self.num_classes,) * 2)

    def compute(self, pred, target):
        mask = (target >= 0) & (target < self.num_classes)

        self.current_state = np.bincount(
            self.num_classes * target[mask].astype('int') + pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return self.current_state

    def update(self, n=1):
        self.cfsmtx += self.current_state

    def accumulate(self):
        accumulate_state = {
            'confusion matrix': self.cfsmtx
        }
        return accumulate_state


class MultiLabelConfusionMatrix(BaseMetric):
    """
    Calculate confusion matrix for multi label segmentation
    Args:
        num_classes (int): number of classes.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.binary = 2
        self.current_state = np.zeros(
            (self.num_classes, self.binary, self.binary))
        super().__init__()

    @staticmethod
    def _check_match(pred, target):
        assert pred.shape == target.shape, \
            "pred should habe same shape with target"

    def reset(self):
        self.cfsmtx = np.zeros((self.num_classes, self.binary, self.binary))

    def compute(self, pred, target):
        mask = (target >= 0) & (target < self.binary)
        for i in range(self.num_classes):
            pred_index_sub = pred[:, i, :, :]
            target_sub = target[:, i, :, :]
            mask_sub = mask[:, i, :, :]
            self.current_state[i, :, :] = np.bincount(
                self.binary * target_sub[mask_sub].astype('int') +
                pred_index_sub[mask_sub], minlength=self.binary ** 2
            ).reshape(self.binary, self.binary)
        return self.current_state

    def update(self, n=1):
        self.cfsmtx += self.current_state

    def accumulate(self):
        accumulate_state = {
            'confusion matrix': self.cfsmtx
        }
        return accumulate_state


@METRICS.register_module
class Accuracy(ConfusionMatrix):
    """
    Calculate accuracy based on confusion matrix for segmentation
    Args:
        num_classes (int): number of classes.
        average (str): {'pixel', 'class'}
            'pixel':
                calculate pixel wise average accuracy
            'class':
                calculate class wise average accuracy
    """

    def __init__(self, num_classes, average='pixel'):
        self.num_classes = num_classes
        self.average = average
        super().__init__(num_classes=self.num_classes)

    def accumulate(self):

        assert self.average in ('pixel', 'class'), \
            'Accuracy only support "pixel" & "class" wise average'

        if self.average == 'pixel':
            accuracy = self.cfsmtx.diagonal().sum() / (
                    self.cfsmtx.sum() + 1e-15)

        elif self.average == 'class':
            accuracy_class = self.cfsmtx.diagonal() / self.cfsmtx.sum(axis=1)
            accuracy = np.nanmean(accuracy_class)

        accumulate_state = {
            'accuracy': accuracy
        }
        return accumulate_state


@METRICS.register_module
class MultiLabelIoU(MultiLabelConfusionMatrix):
    def __init__(self, num_classes):
        super().__init__(num_classes)

    def accumulate(self):
        ious = self.cfsmtx.diagonal(axis1=1, axis2=2) / (
                self.cfsmtx.sum(axis=1) + self.cfsmtx.sum(axis=2) -
                self.cfsmtx.diagonal(axis1=1, axis2=2) + np.finfo(
            np.float32).eps)

        accumulate_state = {
            'IoUs': ious[:, 1]
        }
        return accumulate_state


@METRICS.register_module
class MultiLabelMIoU(MultiLabelIoU):
    def __init__(self, num_classes):
        super().__init__(num_classes)

    def accumulate(self):
        ious = (super().accumulate())['IoUs']

        accumulate_state = {
            'mIoU': np.nanmean(ious)
        }
        return accumulate_state


@METRICS.register_module
class IoU(ConfusionMatrix):
    """
    Calculate IoU for each class based on confusion matrix for segmentation
    Args:
        num_classes (int): number of classes.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(num_classes=self.num_classes)

    def accumulate(self):
        ious = self.cfsmtx.diagonal() / (
                self.cfsmtx.sum(axis=0) + self.cfsmtx.sum(axis=1) -
                self.cfsmtx.diagonal() + np.finfo(np.float32).eps)
        accumulate_state = {
            'IoUs': ious
        }
        return accumulate_state


@METRICS.register_module
class MIoU(IoU):
    """
    Calculate mIoU based on confusion matrix for segmentation
    Args:
        num_classes (int): number of classes.
        average (str): {'equal', 'frequency_weighted'}
            'equal':
                calculate mIoU in an equal class wise average manner
            'frequency_weighted':
                calculate mIoU in an frequency weighted class wise average manner
    """

    def __init__(self, num_classes, average='equal'):
        self.num_classes = num_classes
        self.average = average
        super().__init__(num_classes=self.num_classes)

    def accumulate(self):
        assert self.average in ('equal', 'frequency_weighted'), \
            'mIoU only support "equal" & "frequency_weighted" average'

        ious = (super().accumulate())['IoUs']

        if self.average == 'equal':
            miou = np.nanmean(ious)
        elif self.average == 'frequency_weighted':
            pos_freq = self.cfsmtx.sum(axis=1) / self.cfsmtx.sum()
            miou = (pos_freq[pos_freq > 0] * ious[pos_freq > 0]).sum()

        accumulate_state = {
            'mIoU': miou
        }
        return accumulate_state


class DiceScore(ConfusionMatrix):
    """
    Calculate dice score based on confusion matrix for segmentation
    Args:
        num_classes (int): number of classes.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(self.num_classes)

    def accumulate(self):
        dice_score = 2.0 * self.cfsmtx.diagonal() / (self.cfsmtx.sum(axis=1) +
                                                     self.cfsmtx.sum(axis=0) +
                                                     np.finfo(np.float32).eps)

        accumulate_state = {
            'dice_score': dice_score
        }
        return accumulate_state
