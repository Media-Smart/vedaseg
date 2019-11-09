import numpy as np
import torch.nn.functional as F
import torch


def dice_score(pred, gt, thres_range=np.arange(0.1, 1.0, 0.1)):
    """dice_score
        
        Args:
            pred, n*c*h*w, torch.Tensor
            gt, n*c*h*w, torch.Tensor

        Return:
            dice, nthres * nclasses
    """
    pred = F.interpolate(pred, gt.shape[2:])  #* 0
    gt = gt.float()

    dices = []
    for thres in thres_range:
        tpred = (pred > thres).float()
        nu = 2 * (tpred * gt).sum(dim=[2, 3])
        de = tpred.sum(dim=[2, 3]) + gt.sum(dim=[2, 3])
        dice = nu / de
        dice[torch.isnan(dice)] = 1
        dices.append(dice.sum(0))
    return torch.stack(dices, 0)
