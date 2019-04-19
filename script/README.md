# Ultrasound Nerve Segmentation

* Input: Ultrasound Nerva & Muscle images
* Output: Segemented BP Nerve Images
* Goals: Segment nerves (Brachial Plexus, BP) in ultrasound images.

----
### Training & Testing:
* ```bash train_flow.sh Unet_v7```

```bash
# Copy model configs from Unet_configs/
cp Unet_configs/Unet_v7_config.json ./

# Train model
python3 train.py --id Unet_v7 --config Unet_v7_config.json --data_path ../data --model_path ../models 

# Make Prediction
python3 predict.py --id Unet_v7 --data_path ../data --model_path ../models --pred_path ../preds

# Encode into submission file
python3 submit.py --id Unet_v7 --data_path ../data --output_filepath ../ --pred_path ../preds
```

* Preprocessing:
    * Resize into squared like: (96, 96)
    * Image: Standardization
    * Masks: rescale to [0,1]


* Model:
    * [Unet](https://arxiv.org/abs/1505.04597):
        * Pixel-wise classification
        * Loss: smoothed dice coeff.
        * Data Augmentation
        * Prediction threshold = 0.5
    * [Simple CNN](https://www.kaggle.com/lifepreserver/uns-1-v5-jayjay1-foooork):
        * Predict presence of BP nerve (0/1)
        * If BP nerve is predcited to be present, predict the location of average mask.
    * [Mask-RCNN](https://arxiv.org/pdf/1703.06870.pdf):
        * Modified from [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
        * See [MaskRCNN](./MaskRCNN/README.md)

----
## Improvement:
* Obeservation: False positive instances tends to have small prediction mask.
    * Post-processing or predict presence of Nerve


* Done:
    * Correction of labels (v3)
        * Group similar images by cosine similarity of sub-region of intensity histogram
        * Remove similar images
    * Post-processing: masks with no hole (Morphological close opt)
        * Not much improvement
    * Augmentation: 
        * Elastic transformations did not help much
        * Large rotation/translations prevent learning. (From others)
    * RELU -> ELU, Batch Normalization: help convergence (v4, v5)
    * Change Dense to 1x1 conv: seems also help convergence, but degrades testing performance.


| Method | Val_acc | Val_dice | Public dice | Private dice | Detail | 
|:------:|:------:|:------:|:------:|:------:|:------:|
| Unet_v7 | NA | 0.64 | 0.61733 | 0.64201 | 
| Mask size > Mean(training masks) | NA | 0.732 | 0.6421 | 0.6664 |
| Jointly train SimpleCNN (Sigmoid) (v1) | 0.7927 | 0.6386 | 0.66065 | 0.66693 | score < 0.5 |
| Jointly train SimpleCNN (Softmax) (v2) | 0.8 | 0.6426 | 0.63960 | 0.66858 | score < 0.5 |
| + Cleaned Training Data (v3) | 0.9041 | 0.7289 | 0.63475 | 0.66548 | score < 0.5 <br/>[Starting Code](https://github.com/julienr/kaggle_uns/blob/master/13_clean/0_filter_incoherent_images.ipynb) |
| + RELU->ELU <br/>SimpleCNN: binary loss/sigmoid (v4) | 0.894 | 0.73005 | 0.65724 | 0.65871 |
| + Batch_Normalization <br/>between conv (v5) | 0.8991 | 0.72386 | 0.66125 | 0.66680 | score < 0.5 |
| Change Presence_clf to Conv1x1 (v6) | 0.8831 | 0.726 | 0.60901 | 0.64277 | score < 0.5 |

* Average improvement: 0.64 -> 0.668 (182th -> 103th)
* Highest performance: Average v1~v5 prediction, Private dice = 0.67879 (182th -> 79th)
 

### Further Improvement Thoughts:

* Can try:
    * Prediction threshold and minimum mask size combination
    * Mask-RCNN: use U-net as segmentation


* Some tweaks from others:
    * Inception module, Skip connection with residual blocks
    * Change max-pooling to conv3x3 with stride 2(+BNA ?)
    
    
----
## Applications

#### Model experiment
* Detail results in [Report](Report.pdf)
    * [Quantization](./Quantization/README.md)
    * [Image analysis](./Analysis.ipynb)


#### Model Serving: 
* In [Deployment](./Deployment/README.md)
* Future Work:
    * Improve client.py: Tokens for uploading images

#### Reference:
* [Starting code](https://github.com/jocicmarko/ultrasound-nerve-segmentation)
* http://fhtagn.net/prog/2016/08/19/kaggle-uns.html

