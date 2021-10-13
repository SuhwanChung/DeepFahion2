import torch
import torch.nn as nn
import torch.nn.functional as F

def focalLoss (input, target, weight = None) :
    ce_loss = F.cross_entropy(input, target, weight = weight)
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** 2 * ce_loss).mean()
    return focal_loss