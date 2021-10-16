import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
from flerken.utils.losses import ContrastiveLoss

__all__ = ['BCEWithLogitsLoss', 'MultiTaskLoss']


class BCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, weighted_loss):
        super().__init__()
        self.weighted_loss = weighted_loss

    def forward(self, pred, gt, vs):
        if self.weighted_loss:
            loss = binary_cross_entropy_with_logits(pred, gt, vs['weight'].to(pred.device))
        else:
            loss = binary_cross_entropy_with_logits(pred, gt)
        return loss


class BatchedContrastiveLoss(ContrastiveLoss):
    def forward(self, visual_feat, audio_feat):
        B = visual_feat.size()[0]
        # Normalization step
        x0 = visual_feat / visual_feat.pow(2).sum(dim=2).sqrt().unsqueeze(-1)
        x1 = audio_feat / audio_feat.pow(2).sum(dim=2).sqrt().unsqueeze(-1)

        idx = B // 2
        dist_sync = (x0[:idx] - x1[:idx]).reshape(idx, -1).norm(dim=1)
        dist_usync = (x0[idx:] - x1[idx:][torch.randperm(B - idx)]).reshape(B - idx, -1).norm(dim=1)

        # loss = y * dist_n.pow(2) + (1 - y) * (self.margin - dist_n).clamp(0, self.margin).pow(2)
        loss = torch.cat([dist_sync.pow(2), (self.margin - dist_usync).clamp(0, self.margin).pow(2)])
        if self.weight is not None:
            loss = loss * self.weight
        if self.reduce:
            return loss
        else:
            if self.size_average:
                return torch.mean(loss)
            else:
                return torch.sum(loss)


class MultiTaskLoss(torch.nn.Module):
    def forward(self, pred):
        sep_loss = pred['separation_loss']
        al_loss = pred['alignment_loss']
        if pred['alignment_loss'] is None:
            return sep_loss

        coef = sep_loss.new_tensor([1, 0.1])
        weighted_loss = coef * torch.stack([sep_loss, al_loss])
        pred['weighted_loss'] = weighted_loss
        pred['loss_coef'] = coef
        return weighted_loss.sum()
