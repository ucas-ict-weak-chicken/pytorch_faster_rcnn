import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
from utils.config import cfg
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch
import pdb

class ProposalTargetLayer(nn.Module):
    def __init__(self, nclasses):
        super(ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)




    def forward(self, *input):
        pass