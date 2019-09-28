# Preprcossing functions
import numpy as np
import os, pickle
from os.path import join, exists

from skimage.io import imread
from skimage.transform import resize


def ensure_dir(list_of_dir):
    for d in list_of_dir:
        if not exists(d):
            os.makedirs(d)


def preprocess_x(imgs, new_shape=(96, 96)):
    """
    Preprocess loaded images
    Args:
        imgs: numpy array (num_sample, img_shape[0], img_shape[1])
    Returns:
        x_train: numpy array (num_sample,  img_shape[0], img_shape[1])
    """
    x_train = np.ndarray((imgs.shape[0], new_shape[0], new_shape[1]), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        x_train[i] = resize(imgs[i], new_shape, 
                            preserve_range=True, anti_aliasing=True)

    x_train = x_train[..., np.newaxis].astype('float32')
    
    try:
        ## Get training mean and stdev.
        with open('../data/training_stats.txt', 'r') as f:
            line = f.read().strip()
            mu = float(line.split(' ')[0])
            sigma = float(line.split(' ')[-1])
        print("Training Stats: mu={}, sigma={}".format(mu, sigma))
        
    except FileNotFoundError:
        mu = np.mean(x_train)
        sigma = np.std(x_train)
        with open('../data/training_stats.txt', 'w') as f:
            f.write('{} {}'.format(mu, sigma))
    
    x_train -= mu
    x_train /= (sigma + 1e-7)
    return x_train