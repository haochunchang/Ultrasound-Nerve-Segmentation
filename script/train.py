# python3 train.py --id Unet_v1 --config Unet_v1_config.json --data_path ../data --model_path ../models 
## BuiltIn Modules
import sys, os, json, pickle
import time, random, math
import argparse
from os.path import join

## Image processing & Train/valid split
from skimage.transform import resize
from skimage.io import imsave
from sklearn.model_selection import train_test_split

## Keras & Tensorflow related module
import numpy as np
import keras
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard

## Backend Settings
from keras import backend as K
import tensorflow as tf

## Custom modules
import model_build, utils
from keras.preprocessing.image import ImageDataGenerator


def get_session(gpu_fraction=0.1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                           intra_op_parallelism_threads=3,
                                           inter_op_parallelism_threads=3))


def multilabel_flow(gen1, gen2, x, y_seg, y, configs):
    
    gen = gen1.flow(x, y, shuffle=True, 
                    batch_size=configs['batch_size'], seed=configs['seed'])
    mask_gen = gen2.flow(y_seg, y=None, shuffle=True, 
                         batch_size=configs['batch_size'], seed=configs['seed'])

    for (X_batch, Y1_batch), Y2_batch in zip(gen, mask_gen):
        yield X_batch, {"Segment": Y2_batch, "Presence": Y1_batch}
        
        
#===============
# Main Function
#===============
def main(args):
    
    # Load config files to set hyper-parameters
    with open(args.config, 'r') as f:
        configs = json.load(f)
    utils.ensure_dir([ join(args.model_path, configs['id']) ])
    
    try:
        shape = int(configs['image_size'])
        target_shape = (shape, shape)
    except KeyError:
        target_shape = (96, 96)
    val_size = int(5636 * configs['val_size'])
    train_size = 5636 - val_size
    
    # Training data
    try:
        x_train, y_train = utils.load_train_data(args.data_path)
    except FileNotFoundError:
        utils.create_train_data(args.data_path, args.data_path)
        x_train, y_train = utils.load_train_data(args.data_path)

    X_train = utils.preprocess_x(x_train, target_shape)
    Y_train = utils.preprocess(y_train, target_shape)
    Y_presence = utils.to_category(y_train)
    print(X_train.shape, Y_train.shape, Y_presence.shape)
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, 
                                                      test_size=val_size, 
                                                      random_state=configs['seed'])
    x_train, x_val, y_train_isNerve, y_val_isNerve = train_test_split(X_train, Y_presence, 
                                                                      test_size=val_size, 
                                                                      random_state=configs['seed'])
    
    # build model and save architecture
    builder = model_build.ModelBuilder(input_shape=target_shape, output_shape=target_shape)
    model = builder.build_model(kind=configs['id'].split('_')[0])
    print(model.summary())
    with open(join(args.model_path, configs['id'], "model.json"), 'w') as f:
        f.write(model.to_json())

    # Training callbacks
    checkpointer = ModelCheckpoint(filepath=join(args.model_path, configs['id'], "weights.h5"), 
                                   verbose=1, save_best_only=True,
                                   monitor='val_Segment_dice_coef', mode='max')  
    earlystopping = EarlyStopping(monitor='val_Segment_dice_coef',
                                  patience=int(configs['patience']), verbose=1, mode='max')
    csv_logger = CSVLogger(join(args.model_path, configs['id'], 'train_log.csv'), 
                           append=True, separator=',')
    tfboard = TensorBoard(log_dir=join(args.model_path, configs['id'], 'logs'), 
                          batch_size=configs['batch_size'])
        
    # Data Augmentation & Validation split
    mode = "reflect"
    data_gen_args = dict(rotation_range=int(configs["rotation_range"]),
                         shear_range=float(configs["shear"]),
                         zoom_range=float(configs["zoom"]),
                         width_shift_range=float(configs["width_shift_range"]),
                         height_shift_range=float(configs["height_shift_range"]),
                         horizontal_flip=bool(configs["horizontal_flip"]),
                         vertical_flip=bool(configs["horizontal_flip"]),
                         fill_mode=mode
                        )
    seed = configs['seed']
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(x_train, augment=True, seed=seed)
    mask_datagen.fit(y_train, augment=True, seed=seed)
    
    val_img_datagen = ImageDataGenerator(fill_mode=mode)
    val_mask_datagen = ImageDataGenerator(fill_mode=mode)
    val_img_datagen.fit(x_val, augment=True, seed=seed)
    val_mask_datagen.fit(y_val, augment=True, seed=seed)
    
    X_train_augmented = multilabel_flow(image_datagen, mask_datagen, 
                                        x_train, y_train, y_train_isNerve, configs)
    #Y_train_augmented = mask_datagen.flow(y_train, batch_size=configs['batch_size'], shuffle=True, seed=seed)
    X_val_augmented = multilabel_flow(val_img_datagen, val_mask_datagen, 
                                      x_val, y_val, y_val_isNerve, configs)
    #Y_val_augmented = multilabel_flow(val_mask_datagen, y_val, None, configs)
    
    train_generator = X_train_augmented#zip(X_train_augmented, Y_train_augmented)
    val_generator = X_val_augmented#zip(X_val_augmented, Y_val_augmented)
    
    model.fit_generator(train_generator,
                        epochs=configs['epoch'],
                        validation_data=val_generator, 
                        validation_steps=(val_size // configs['batch_size']),
                        steps_per_epoch=(train_size // configs['batch_size']),
                        callbacks=[earlystopping, checkpointer, csv_logger, tfboard]
                       )
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True, help="model id")
    parser.add_argument("--config", type=str, default='./Unet_v1_config.json', help="filepath of hyper parameter config")
    parser.add_argument("--data_path", type=str, default='../data', help="filepath of train/dev data set")
    parser.add_argument("--model_path", type=str, default='../models', help="filepath of trained models")
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    #K.set_session(get_session())
    K.set_image_dim_ordering('tf')
    main(args)
