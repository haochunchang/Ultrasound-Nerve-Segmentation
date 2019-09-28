# Ultrasound Nerve Segmentation

* Input: Ultrasound Nerva & Muscle images
* Output: Segemented BP Nerve Images
* Goals: Segment nerves (Brachial Plexus, BP) in ultrasound images.

----
## Model Quantization

```bash
# Convert to pb file
python3 ./Quantization/KtoTF.py --help
python3 ./Quantization/KtoTF.py --input_model ../models/Unet_v7/weights.h5 \
                         --input_model_json ../models/Unet_v7/model.json \
                         --output_model ../Unet_v7/Unet_v7.pb \
                         --quantize

# Get prediction from pb file model
python3 Quantized_model_predict.py --model_dir ../models --model_id Unet_v7
```

* See [Explore.ipynb](Explore.ipynb) for usage example