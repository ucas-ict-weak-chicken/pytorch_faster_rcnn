import argparse
import numpy as np
import torch

def parse_args():
    """
    Parse input arguments.

    Returns:
        Args dict.
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--optimizer', dest='optimizer', help='Optimizer type', default='adam', type=str)
    parser.add_argument('--iters', dest='max_iters', help='Number of iterations to train', default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model', help='Initialize with pretrained model weights', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name', help='dataset to train on', default='voc_2007_trainval', type=str)

    args = parser.parse_args()
    return args

def combined_roidb(imdb_names):
    """

    Args:
        imdb_names:

    Returns:
        imdb, roidb
    """
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training)'.format(imdb.name))
        roidb = get_training_roidb(imdb)
        return roidb

if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    imdb, roidb = combined_roidb(args.imdb_name)


