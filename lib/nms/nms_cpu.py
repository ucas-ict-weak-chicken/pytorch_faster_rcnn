#coding:utf-8
from __future__ import absolute_import

import numpy as np
import torch

def nms_cpu(dets, thresh):
    """
    nms超过一定阈值的候选框中只保留最大的那个
    Args:
        dets(torch.Tensor): [x1, y1, x2, y2, cls_scores]
        thresh: nms thresh

    Returns:
        torch.Tensor: 需要保留的框的下标。

    """
    dets = dets.numpy()
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = 1.0 * inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1] #因为ovr是从第二个元素开始的，因此要+1

    return torch.IntTensor(keep)
