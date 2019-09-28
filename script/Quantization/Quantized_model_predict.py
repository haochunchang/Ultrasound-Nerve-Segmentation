"""
Get prediction from Unet models stored as tensorflow pb file
"""

import tensorflow as tf
import numpy as np
from os.path import join

import utils


def predict_pb(model_dir, model_id):

    imgs_test, imgs_id_test = utils.load_test_data('../data')
    target_shape = (128,128)
    imgs_test = utils.preprocess_x(imgs_test, new_shape=target_shape)

    bz = 32
    preds = np.zeros_like(imgs_test, dtype=np.float64)
    model_path = join(model_dir, model_id, '{}.pb'.format(model_id))

    with tf.Session() as sess:
        with tf.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="Unet")

        input_tensor_name = "Unet/input_1:0"
        output_tensor_name = "Unet/conv2d_19_1/Sigmoid:0"
        #x = graph.get_tensor_by_name(input_tensor_name)
        #y = graph.get_tensor_by_name(output_tensor_name)
        for i in range(imgs_test.shape[0] // bz):
            pred = sess.run(output_tensor_name, 
                             feed_dict={input_tensor_name: imgs_test[i*bz:(i+1)*bz]})
            preds[i*bz:(i+1)*bz] = pred
            end = i
        # Last remaining data
        pred = sess.run(output_tensor_name, 
                        feed_dict={input_tensor_name: imgs_test[(end+1)*bz:]})
        preds[(end+1)*bz:] = pred

    print(preds.shape)
    np.save(join(model_id, '{}_imgs_mask_test.npy'.format(model_id)), preds)
    
    
if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default='../models',
                        help="Directory of models")
    parser.add_argument('--model_id', type=str, required=True,
                        help='Model ID')
    
    args = parser.parse_args()
    predict_pb(args.model_dir, args.model_id)