import torch
import numpy as np
from torch.utils.cpp_extension import load
import time

import nms.cuda_nms as nms

def nms_gpu(dets, thresh):
    scores = dets[:, 4]
    num_boxes, boxes_dim = dets.size()
    keep = dets.new(dets.size(0), 1).zero_().int()
    num_out = dets.new(1).zero_().int()

    _, order = torch.sort(scores, 0, True)
    sorted_dets = dets[order]

    nms.nms_cuda(keep, num_out, sorted_dets, num_boxes, boxes_dim, thresh)
    keep = keep[:num_out[0]][:, 0].long()
    return order[keep]


if __name__ == '__main__':
    dets = torch.tensor([
        [1, 1, 100, 100, 1],
        [5, 1, 100, 100, 0.9],
        [10, 1, 100, 100, 0.99],
        [100, 100, 300, 300, 0.93],
        [100, 100, 300, 400, 0.97],
        [500, 1, 600, 100, 0.95],
    ]).float().cuda()
    print(dets)
    thresh = 0.1
    t = time.time()
    print(nms_gpu(dets, thresh))
    from IPython import embed
    embed()
