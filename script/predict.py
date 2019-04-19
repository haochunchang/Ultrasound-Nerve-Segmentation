# python3 predict.py --id Unet_v1--data_path ../data --model_path ../models --pred_path ../preds
from os.path import join
import os
import argparse
import numpy as np
from skimage.io import imsave

## Backend Settings
from keras import backend as K
import tensorflow as tf
def get_session(gpu_fraction=0.2):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

os.environ["CUDA_VISIBLE_DEVICES"]="0"
#K.set_session(get_session())
K.set_image_dim_ordering('tf')

import model_build, utils


def main(args):
    
    utils.ensure_dir([ join(args.model_path, args.id), 
                       join(args.pred_path, args.id) ])
    
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    try:
        imgs_test, imgs_id_test = utils.load_test_data(args.data_path)
    except FileNotFoundError:
        utils.create_test_data(args.data_path, args.data_path)
        imgs_test, imgs_id_test = utils.load_test_data(args.data_path)
        
    print('-'*30)
    print('Loading saved model...')
    print('-'*30)
    net = model_build.load_model(join(args.model_path, args.id))

    output_shape = net.layers[0].output_shape[1:-1]
    if len(output_shape) == 2:
        imgs_test = utils.preprocess_x(imgs_test, new_shape=output_shape)
    else:
        imgs_test = utils.preprocess_x(imgs_test, new_shape=(96, 96))
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test, presence_test = net.predict(imgs_test, verbose=1)
    print(imgs_mask_test.shape, presence_test.shape)
    np.save(join(args.pred_path, args.id, 'imgs_mask_test.npy'), imgs_mask_test)
    np.save(join(args.pred_path, args.id, 'imgs_presence_test.npy'), presence_test)

    """print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(join(args.pred_path, args.id, str(image_id) + '_pred.png'), image)"""
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True, help="model id")
    parser.add_argument("--data_path", type=str, default='./data', help="filepath of train/dev data set")
    parser.add_argument("--model_path", type=str, default='./model', help="filepath of trained models")
    parser.add_argument("--pred_path", type=str, default='./output', help="filepath of output predictions")
    
    args = parser.parse_args()
    main(args)
