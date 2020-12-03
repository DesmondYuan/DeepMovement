import numpy as np
from PIL import Image
import hashlib

path = "/static/img"


def get_pil_array(img_arr, res=32):
    img = np.array(img_arr)
    # img = img.reshape(img.shape[0], img.shape[1], 4)
    return img


def main_predict(img_content, img_style):
    # placeholder function for Magenta
    img1 = get_pil_array(img_content)
    img2 = get_pil_array(img_style)
    output = (img1+img2)/2
    outfns = save_img(img1), save_img(img2), save_img(output)
    return outfns

def save_img(img):
    outfn = md5({'fingerprint': img[:,0,0]}) + '.png'
    im = Image.fromarray(img)
    im.save(outfn)


def md5(obj):
    key = json.dumps(vars(obj), sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()
