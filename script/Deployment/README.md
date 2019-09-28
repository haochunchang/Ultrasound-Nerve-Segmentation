# Ultrasound Nerve Segmentation

* Input: Ultrasound Nerva & Muscle images
* Output: Segemented BP Nerve Images
* Goals: Segment nerves (Brachial Plexus, BP) in ultrasound images.

----
## Model Serving

#### Starting development server:

1. Set up docker container:

```bash
docker pull tensorflow/tensorflow:1.10.1-gpu-py3
nvidia-docker run -it -v /path/to/this_dir:/deploy/ \
                  -p 5000:5000 \
                  --name="Nerve_server" \
                  tensorflow/tensorflow:1.10.1-gpu-py3 bash
docker attach Nerve_server
cd ../deploy/
```

2. Start flask server:

```bash
pip3 install -r requirement.txt
python3 server.py --pb_model_filename ./Unet_v7.pb --gpu_memory 0.5 #(default)
# Server will host in 0.0.0.0:5000
```

#### Use client to test

```bash
source Test_Nerve/bin/activate
python3 client.py --url 127.0.0.1:5000/api/predict \
                  --img_path ../data/x_train/images/11_1.tif
```

        
    
