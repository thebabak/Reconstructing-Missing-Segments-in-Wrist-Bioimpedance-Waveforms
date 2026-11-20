"""Masked MAE from the upstream MIT-licensed SAITS utilities; see LICENSE."""
import torch

def masked_mae_cal(inputs, target, mask):
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)
