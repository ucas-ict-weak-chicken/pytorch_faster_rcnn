from torchvision.models import vgg16
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes
from nms.nms_wrapper import nms


class ProposalLayer(nn.Module):
    def __init__(self, feat_stride, scales, ratios):
        super(ProposalLayer, self).__init__()
        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales), ratios=np.array(ratios))).float() #type: torch.Tensor # N*4
        self._num_anchors = self._anchors.size(0)



    def forward(self, scores, bbox_deltas, im_info, cfg_key):
        """
        1. 生成Anchor Box
        2. 将bbox预测应用到不同Anchor Box上，并对超出图片的部分做clipping，生成proposals
        3. 根据scores对proposals做nms，最终只保留阈值超过thresh且经过nms后的固定数量的proposals。

        Args:
            scores(torch.Tensor): [B 2A H W] The output of object score conv layer.
            bbox_deltas(torch.Tensor): [B 4A H W] The output of bbox conv layer
            im_info(torch.Tensor): [B, 2] Widths and heights of img
            cfg_key: TRAIN or TEST, will influence nms thresh

        Returns:
            torch.Tensor: [B, post_nms_topN, 5], the last dim represent [batch_id, x1, y1, x2, y2]
        """
        scores = scores[:, self._num_anchors:, :, :] #type: torch.Tensor

        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE

        batch_size = bbox_deltas.size(0)
        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y) #type:np.ndarray, np.ndarray
        shifts = torch.from_numpy(np.vstack([shift_x.ravel(),
                                            shift_y.ravel(),
                                            shift_x.ravel(),
                                            shift_y.ravel()]).transpose()) #type: torch.Tensor

        shifts = shifts.contiguous().type_as(scores).float()
        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(scores) #type: torch.Tensor
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K*A, 4).expand(batch_size, K*A, 4)

        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        scores = scores.permute(0, 2, 3, 1).contiguous()
        scores = scores.view(batch_size, -1)

        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)

        proposals = clip_boxes(proposals, im_info, batch_size)

        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new(batch_size, post_nms_topN, 5).zero_()

        for i in range(batch_size):
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1, 1)

            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
                proposals_single = proposals_single[keep_idx_i, :]
                scores_single = scores_single[keep_idx_i, :]

                num_proposal = proposals_single.size(0)
                output[i, :, 0] = i
                output[i, :num_proposal, 1:] = proposals_single

        return output





class AnchorTargetLayer(nn.Module):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales, ratios):
        super(AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        self._allowed_border = 0


    def forward(self, rpn_cls_score, gt_boxes, im_info, num_boxes):
        """
        for each (H, W) location i
            generate 9 anchor boxes centered on cell i
            apply predicted bbox deltas at cell i to each of the 9 anchors
        filter out-of-image anchors
        Args:
            rpn_cls_score(torch.Tensor): [B, 2, H, W]
            gt_boxes:
            im_info:
            num_boxes:

        Returns:

        """
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        batch_size = gt_boxes.size(0)

        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        # 生成anchors
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(gt_boxes)  # move to specific gpu.
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K * A, 4)

        total_anchors = int(K*A)

        # clip anchors
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < int(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < int(im_info[0][0]) + self._allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive 0 is negtive -1 is dont care
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()

        overlaps = bbox_overlaps_batch(anchors, gt_boxes)



class RPN(nn.Module):
    def __init__(self, din, anchor_scales, anchor_ratios, feat_stride):
        """
        RPN网络。接受基础网络传来的feature map，返回feature map每个位置的预测框。

        Args:
            din(int): input dimension
            anchor_scales(list[float]): scales for anchor box
            anchor_ratios(list[float]): aspect ratios of anchor box
            feat_stride(int): feature strides
        """
        super(RPN, self).__init__()
        self.din = din
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.feat_stride = feat_stride

        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # bg/fg
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        self.RPN_proposal = ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.RPN_anchor_target = AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        """
        Reshape x from [B, C, H, W] to [B, d, C*H/d, W]

        Args:
            x(torch.Tensor): [B, C, H, W], input tensor
            d(int): target d

        Returns:
            torch.Tensor: tensor after reshape

        """
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x


    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
        """

        Args:
            base_feat(torch.Tensor): [B, C, H, W],feature map of basenet
            im_info(torch.Tensor): [B, 2] width and height of image.
            gt_boxes:
            num_boxes:

        Returns:

        """
        batch_size = base_feat.size(0)

        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        # 将分类结果输出转化为概率
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = self.RPN_proposal(rpn_cls_prob.data, rpn_bbox_pred.data, im_info, cfg_key)  # type: torch.Tensor

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None
            rpn_data = self.RPN_anchor_target(rpn_cls_score.data, gt_boxes, im_info, num_boxes)