import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from utils.config import cfg
from rpn.rpn import RPN
from roi_pooling.modules.roi_pool import _RoIPooling
from roi_crop.modules.roi_crop import _RoICrop
from roi_align.modules.roi_align import RoIAlignAvg
from rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
from abc import abstractmethod

class FasterRCNN(nn.Module):
    def __init__(self, classes, dout_base_model, class_agnostic):
        super(FasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.dout_base_model = dout_base_model


    def init_model(self):
        # define rpn
        self.RCNN_rpn = RPN(self.dout_base_model, cfg.ANCHOR_SCALES, cfg.ANCHOR_RATIOS, cfg.FEAT_STRIDE[0])
        self.RCNN_proposal_target = ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()
