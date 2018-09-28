import torch
from model.utils.config import cfg
if torch.cuda.is_available():
    from model.nms.nms_gpu import nms_gpu
from model.nms.nms_cpu import nms_cpu

def nms(dets, thresh, force_cpu=Fasle):
    if dets.shape[0] == 0:
        return []
    return nms_gpu(dets, thresh) if force_cpu == False else nms_cpu(dets, thresh)