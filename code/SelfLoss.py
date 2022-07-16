import torch
import torch.nn as nn


class SimilarityMarginLoss(nn.Module):
    def __init__(self):
        super(SimilarityMarginLoss, self).__init__()
        self.l1_loss = nn.SmoothL1Loss()

    def forward(self, predict, label):
        loss = torch.clamp(predict - label, min=0).mean()
        # loss = torch.clamp(torch.abs(predict - label) - 0.1, min=0).mean()
        # print(predict.mean(), label.mean(), loss.mean())
        return loss


        # abs(0.7 - 0.9) = - 0.2 - 0.1 = 0.1
        # abs(0.89 - 0.9) = 0.01 - 0.1 =
