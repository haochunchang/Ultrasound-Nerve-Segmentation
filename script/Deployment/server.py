import json, argparse, time, base64
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import utils


from flask import Flask, request
from flask_cors import CORS


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
        
    return graph


##################################################
# API part
##################################################
app = Flask(__name__)
cors = CORS(app)

@app.route("/api/predict", methods=['POST'])
def predict():
    start = time.time()
    
    target_shape = (128,128)
    data = request.data
    data = base64.b64decode(data)
    
    x_in = np.frombuffer(data, dtype='uint8').reshape((420, 580))
    x_in = x_in.reshape((1,) + x_in.shape)
    x_in = utils.preprocess_x(x_in, target_shape)
    
    ##################################################
    # Tensorflow part
    ##################################################
    y_out = persistent_sess.run(y, 
                                feed_dict={ x: x_in }
                               )
    ##################################################
    # END Tensorflow part
    ##################################################
    
    json_data = json.dumps({'y': y_out.tolist()})
    print("Time spent handling the request: %.3f seconds" % (time.time() - start))
    
    return json_data
##################################################
# END API part
##################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb_model_filename", default="../../Unet_v7/Unet_v7.pb", type=str, help="Frozen model file to import (.pb)")
    parser.add_argument("--gpu_memory", default=.5, type=float, help="GPU memory per process")
    args = parser.parse_args()

    ##################################################
    # Tensorflow part
    ##################################################
    print('Loading the model')
    graph = load_graph(args.pb_model_filename)
    x = graph.get_tensor_by_name('prefix/input_1:0')
    y = graph.get_tensor_by_name('prefix/conv2d_19_1/Sigmoid:0')

    print('Starting Session, setting the GPU memory usage to %f' % args.gpu_memory)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    persistent_sess = tf.Session(graph=graph, config=sess_config)
    ##################################################
    # END Tensorflow part
    ##################################################

    print('Starting the Server')
    app.run(host="0.0.0.0", port=5000)