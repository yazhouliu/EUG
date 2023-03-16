import torch
import torch.nn as nn
import torch.nn.functional as F

class SegLoss(object):
    def __init__(self, loss_cfg, **kwargs):
        self.ignore_label = loss_cfg.IGNORE_LABEL
        self.nll_loss = nn.NLLLoss(ignore_index=self.ignore_label, reduction="mean")

    def __call__(self, output, target):
        log_softmax = F.log_softmax(output["binary_segmentation"], dim=1)
        xent_loss = self.nll_loss(log_softmax, target.long())

        return xent_loss
