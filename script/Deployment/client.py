import requests
import json, time, pickle
from os.path import join
from skimage.io import imread
import base64

headers = requests.utils.default_headers()
headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) Chrome/56.0.2924.87 Safari/537.36'
headers['content-type'] = 'image/tif'

def send_image_request(img_path, url):
    
     # read image
    img = imread(img_path, as_gray=True, dtype='uint8')
    data = base64.b64encode(img.tobytes())
    data = data.decode('utf-8')

    # Send POST request
    print("Sending request to {}".format(url))
    response = requests.post(url, 
                             data=data,
                             headers=headers)
    time.sleep(0.001)
    return response


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5000/api/predict", 
                        type=str, help="URL of serving segmentation model")
    parser.add_argument("--img_path", default="join('..', 'data', 'x_train', 'images', '11_1.tif')", 
                        type=str, help="File path of input image (.tif)")
    args = parser.parse_args()
    
    #=======
    # Main
    #=======
    response = send_image_request(args.img_path, args.url)
    with open('response.pkl', 'wb') as f:
        pickle.dump(response, f)
    
