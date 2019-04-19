# Preprcossing functions
import numpy as np
import os, pickle
from os.path import join, exists

from skimage.io import imread
from skimage.transform import resize
from skimage.filters import sobel
from skimage.exposure import equalize_hist

from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

def ensure_dir(list_of_dir):
    for d in list_of_dir:
        if not exists(d):
            os.makedirs(d)

            
#====================
# Compute statistcs
#====================
def get_train_stats(data_path):
    try:
        with open(join(data_path, 'training_stats.pkl'), 'rb') as f:
            stats = pickle.load(f)

    except FileNotFoundError:
        print('Loading')
        imgs, _ = load_train_data(data_path)
        print('Computing')
        mean = imgs.mean(axis=0)
        std = imgs.std(axis=0)
        stats = {}
        stats['mean'] = mean
        stats['std'] = std
        with open(join(data_path, 'training_stats.pkl'), 'wb') as f:
            pickle.dump(stats, f)

    return stats['mean'], stats['std']
            
    
#=====================
# Load Data
#=====================
def create_train_data(data_path, out_path):
    
    real_data_path = join(data_path, 'x_train', 'images')
    real_mask_path = join(data_path, 'y_train', 'images')
    images = os.listdir(real_data_path)
    masks = os.listdir(real_mask_path)
    total = len(images)
    assert len(images) == len(masks)

    img = imread(join(real_data_path, images[0]), as_gray=True)
    image_rows, image_cols = img.shape
    
    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint16)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint16)

    print('-'*30)
    print('Creating images...')
    print('-'*30)
    for i, image_name in enumerate(images):
        
        img = imread(join(real_data_path, image_name), as_gray=True)
        img = np.array([img])
        imgs[i] = img
        
        mask_name = image_name.strip('.tif') + '_mask.tif'
        mask = imread(join(real_mask_path, mask_name), as_gray=True)
        mask = np.array([mask])
        imgs_mask[i] = mask
        
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
    print('Loading done.')

    np.save(join(out_path, 'imgs_train.npy'), imgs)
    np.save(join(out_path, 'imgs_mask_train.npy'), imgs_mask)
    print('Saving to .npy files done.')
    

def load_train_data(data_path):
    imgs = np.load(join(data_path, 'imgs_train.npy'))
    imgs_mask = np.load(join(data_path, 'imgs_mask_train.npy'))
    return imgs, imgs_mask


def create_test_data(data_path, out_path):
    
    real_data_path = join(data_path, 'test')
    images = os.listdir(real_data_path)
    total = len(images)

    img = imread(join(real_data_path, images[0]), as_gray=True)
    image_rows, image_cols = img.shape
    
    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        imgs_id[i] = img_id
            
        img = imread(join(real_data_path, image_name), as_gray=True)
        img = np.array([img])
        imgs[i] = img
        
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(join(out_path, 'imgs_test.npy'), imgs)
    np.save(join(out_path, 'imgs_id_test.npy'), imgs_id)
    print('Saving to .npy files done.')


def load_test_data(data_path):
    imgs = np.load(join(data_path, 'imgs_test.npy'))
    imgs_mask = np.load(join(data_path, 'imgs_id_test.npy'))
    return imgs, imgs_mask


#======================
# Preprocessing
#======================
def preprocess(imgs, new_shape=(96, 96)):
    """
    Reshape and scale images
    Args:
        imgs: numpy array (num_sample, img_shape[0], img_shape[1])
    Returns:
        imgs_p: numpy array (num_sample,  img_shape[0], img_shape[1])
    """
    imgs_p = np.ndarray((imgs.shape[0], new_shape[0], new_shape[1]), dtype=np.uint16)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], new_shape, mode='reflect',
                           preserve_range=True, anti_aliasing=True)

    imgs_p = imgs_p[..., np.newaxis].astype('float32') / 255.0
    return imgs_p


def preprocess_x(imgs, new_shape=(96, 96)):
    """
    Reshape and standardized images by mean, std from training images
    Args:
        imgs: numpy array (num_sample, img_shape[0], img_shape[1])
    Returns:
        x_train: numpy array (num_sample,  img_shape[0], img_shape[1])
    """
    x_train = np.ndarray((imgs.shape[0], new_shape[0], new_shape[1]), dtype=np.uint16)
    for i in range(imgs.shape[0]):
        x_train[i] = resize(imgs[i], new_shape, mode='reflect',
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


def to_category(y, num_class=2):
    out = []
    for i in range(len(y)):
        if y[i].sum() == 0:
            out.append(0)
        else:
            out.append(1)
    out = np.array(out)
    return to_categorical(out, num_class)