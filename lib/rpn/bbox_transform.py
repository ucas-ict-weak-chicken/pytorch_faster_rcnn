import torch
import numpy as np


def bbox_transform_batch(anchors, gt_boxes):
    """
    ground_truth和锚框之间的IoU

    Args:
        anchors(torch.Tensor): [B, N, 4] or [N, 4] 锚框
        gt_boxes(torch.Tensor): [B, K, 4] 真实框

    Returns:
        torch.Tensor: (N, K) ndarray of overlap between boxes and query boxes

    """
    batch_size = gt_boxes.size(0)

    if anchors.dim() == 2:

        N = anchors.size(0)
        K = gt_boxes.size(1)

        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:, :, :4].contiguous()

        gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
        gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
        anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, 1, K)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) - torch.max(boxes[:, :, :, 0],
                                                                               query_boxes[:, :, :, 0]) + 1
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
              torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)

        if anchors.size(2) == 4:
            anchors = anchors[:,:,:4].contiguous()
        else:
            anchors = anchors[:,:,1:5].contiguous()

        gt_boxes = gt_boxes[:,:,:4].contiguous()

        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)

        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps


def bbox_transform_inv(boxes, deltas, batch_size):
    """
    将神经网络输出的预测转化到实际的锚框上

    Args:
        boxes(torch.Tensor): [B, HWA, 4] anchor box
        deltas(torch.Tensor): [B, HWA, 4] the output of bbox conv
        batch_size(int): batch size


    Returns:
        torch.Tensor: [B, HWA, 4] 转化后的预测框
    """
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0  # type: torch.Tensor
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0  # type: torch.Tensor
    ctr_x = boxes[:, :, 0] + 0.5 * widths  # type: torch.Tensor
    ctr_y = boxes[:, :, 1] + 0.5 * heights  # type: torch.Tensor

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape, batch_size):
    """
    将超出图像大小的框改为合适的框（会修改boxes）
    Args:
        boxes(torch.Tensor): (B, K, 4)
        im_shape(torch.Tensor): (B, 2)
        batch_size(int): batch size

    Returns:
        torch.Tensor: 修改后的框
    """
    for i in range(batch_size):
        boxes[i, :, 0::4].clamp_(0, im_shape[i, 1] - 1)
        boxes[i, :, 1::4].clamp_(0, im_shape[i, 0] - 1)
        boxes[i, :, 2::4].clamp_(0, im_shape[i, 1] - 1)
        boxes[i, :, 3::4].clamp_(0, im_shape[i, 0] - 1)
    return boxes
