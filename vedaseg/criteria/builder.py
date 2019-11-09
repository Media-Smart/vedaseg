from .seg_wrapper import CriterionWrapper


def build_criterion(cfg):
    criterion = CriterionWrapper(cfg)
    return criterion
