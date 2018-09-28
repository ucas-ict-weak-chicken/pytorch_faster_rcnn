import torch
from torch.optim import SGD

def stage1_rpn_train(model:torch.nn.Module):
    return SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)