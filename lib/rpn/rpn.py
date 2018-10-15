from torchvision.models import vgg16
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
from utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, bbox_overlaps_batch, bbox_transform_batch
from nms.nms_wrapper import nms
from utils.net_utils import _smooth_l1_loss

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
            gt_boxes(torch.Tensor): [B, K, 4] Where K is the max box of an image
            im_info(torch.Tensor): [B, 2] width and height of image
            num_boxes(int): num of boxes

        Returns:
            list[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: labels(B, 1, AH, W), bbox target(B, A, H, W), inside weights(B, A, H, W), outside weights(B, A, H, W)
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

        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)  # 锚框对真实框取最大
        gt_max_overlaps, _ = torch.max(overlaps, 1)  # 真实框对锚框取最大


        # 如果映射到锚框上的IoU最大的gt依然小于阈值，对应锚框label为0
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0



        gt_max_overlaps[gt_max_overlaps==0] = 1e-5

        # 一个锚框对应的最大真实框的数量
        # TODO: 这种方式效率太低了吧
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)
        if torch.sum(keep) > 0:
            labels[keep>0] = 1

        # 锚框对应的最大真实框大于一定阈值
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        for i in range(batch_size):
            # 忽略多余的正例
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # 忽略多余的负例
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0)))
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1

        offset = torch.arange(0, batch_size) * gt_boxes.size(1)

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)

        # [B, A, 4] regression box.
        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

        bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert (
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1)
            )

        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights

        # 去除图片外的边框
        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=0)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []

        labels = labels.view(batch_size, height, width, A).permute(0, 3, 1, 2).contiguous()

        #TODO: Check why
        labels = labels.view(batch_size, 1, A*height, width)
        outputs.append(labels)

        bbox_targets = bbox_targets.view(batch_size, height, width, A).permute(0, 3, 1, 2).contiguous()
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size, anchors_count, 1).expand(batch_size, anchors_count, 4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
                                .permute(0,3,1,2).contiguous()
        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        outputs.append(bbox_outside_weights)


        return outputs

    def backward(self):
        pass




def _compute_targets_batch(ex_rois, gt_rois):
    """
    将真实框转化成回归框
    Args:
        ex_rois(torch.Tensor): [A, 4] anchors
        gt_rois(torch.Tensor): [B, A, 5] max gt boxes

    Returns:
        torch.Tensor: [B, A, 4]
    """
    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])


def _unmap(data, count, inds, batch_size, fill=0):
    """
    取出data[inds]到内存中，不够的补0.

    Args:
        data(torch.Tensor): 原始数据，二维或三维
        count(int): 切片后最大长度
        inds(torch.Tensor): 下标信息
        batch_size(int): bs
        fill(int): 不够长度补fill

    Returns:
        torch.Tensor: [B, count, *]
    """
    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds, :] = data
    return ret





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
            gt_boxes(torch.Tensor): [B, K, 4] Where K is the max box of an image
            num_boxes(int):

        Returns:
            rois, rpn_loss_cls and rpn_loss_box
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

            # rpn_cls_score只用了形状信息
            rpn_data = self.RPN_anchor_target(rpn_cls_score.data, gt_boxes, im_info, num_boxes)

            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

            return rois, self.rpn_loss_cls, self.rpn_loss_box