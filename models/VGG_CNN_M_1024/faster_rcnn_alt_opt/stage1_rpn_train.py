import torch
from torch.optim import SGD

def getOptimizer(model:torch.nn.Module):
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)

