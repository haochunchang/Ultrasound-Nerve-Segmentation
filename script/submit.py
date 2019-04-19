"""
Combine multiple network predictions
"""

import numpy as np
from os.path import join
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage.morphology import closing
import glob, datetime

import utils
from utils import load_test_data


def find_best_mask(data_path):
    """
    Find mask from all training data masks.
    """
    files = glob.glob(join(data_path, "*_mask.tif"))
    overall_mask = imread(files[0], as_gray=True)
    overall_mask.fill(0)
    overall_mask = overall_mask.astype(np.float32)

    for fl in files:
        mask = imread(fl, as_gray=True)
        overall_mask += mask
    overall_mask /= 255.0
    max_value = overall_mask.max()
    koeff = 0.5
    overall_mask[overall_mask < koeff * max_value] = 0
    overall_mask[overall_mask >= koeff * max_value] = 255
    overall_mask = overall_mask.astype(np.uint8)
    imsave("Overall_mask.png", overall_mask)
    return overall_mask


def prep(img, shape):
    img = img.astype('float32')
    img = (img > 0.5).astype(np.uint8)  # threshold
    img = resize(img, shape, preserve_range=True)
    return img


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def submit_Simple(data_path, output_filepath, pred_path, model_id='SimpleCNN_v1'):
    
    imgs_test, imgs_id_test = load_test_data(data_path)
    imgs_test = np.load(join(pred_path, model_id, 'imgs_mask_test.npy'))
    
    argsort = np.argsort(imgs_id_test)
    test_id = imgs_id_test[argsort]
    predictions = imgs_test[argsort]
    
    sub_file = join('submission_simple_' + str(datetime.datetime.now().strftime("%m-%d-%H")) + '.csv')
    subm = open(sub_file, "w")
    
    mask = find_best_mask(join(data_path, "y_train", "images"))
    encode = run_length_enc(mask)
    subm.write("img,pixels\n")
    for i in range(len(test_id)):
        subm.write(str(test_id[i]) + ',')
        if predictions[i][1] > 0.5:
            subm.write(encode)
        subm.write('\n')
        
    subm.close()


def submit_Unet(data_path, output_filepath, pred_path, model_id=['Unet_v1']):

    imgs_test, imgs_id_test = load_test_data(data_path)
    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    
    predictions = []
    isNerve_scores = []
    for ind in model_id:
        imgs_test = np.load(join(pred_path, ind, 'imgs_mask_test.npy'))
        imgs_test = imgs_test[argsort]
        predictions.append(imgs_test)
        
        pre_test = np.load(join(pred_path, ind, 'imgs_presence_test.npy'))
        pre_test = pre_test[argsort]
        isNerve_scores.append(pre_test)

    total = imgs_test.shape[0]
    ids = []
    rles = []
    for i in range(total):
        
        final_img = np.zeros((420, 580))
        score = 0
        for imgs_test, s in zip(predictions, isNerve_scores):
            img = imgs_test[i, :]
            new_img = resize(img.astype('float32'), (420, 580), 
                             preserve_range=True, anti_aliasing=True, mode='reflect')
            new_img = new_img.reshape((420, 580))
            final_img += new_img
            score += np.argmax(s[i,:])
            
        final_img = final_img / len(predictions)
        score = score / len(predictions)
        final_img = (final_img > 0.5).astype(np.uint8)
        final_img = closing(final_img)
        
        # Filter mask size
        #if final_img.sum() < 2954:
        if score < 0.5:
            final_img = np.zeros((420, 580))
            
        rle = run_length_enc(final_img)
        
        rles.append(rle)
        ids.append(imgs_id_test[i])

        if i % 100 == 0:
            print('{}/{}'.format(i, total))

    first_row = 'img,pixels'
    file_name = join(output_filepath)

    with open(file_name, 'w') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')
    print('Submission Saved.')
     
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True, help="model id")
    parser.add_argument("--data_path", type=str, default='./data', help="filepath of test data set")
    parser.add_argument("--output_filepath", type=str, default='./submission_ensemble.csv', help="filepath of output submission")
    parser.add_argument("--pred_path", type=str, default='./output', help="filepath of output predictions")
    
    args = parser.parse_args()
    #submit_Simple(args.data_path, args.output_filepath, args.pred_path)
    submit_Unet(args.data_path, args.output_filepath, args.pred_path, args.id.split(','))