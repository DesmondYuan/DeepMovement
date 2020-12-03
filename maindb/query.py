import numpy as np
from PIL import Image

path = "/static/img"


def get_pil_array(img_arr, res=32):
    img = img_arr.reshape(img_arr.shape[0], img_arr.shape[1], 3)
    return img


def main_predict(img_content, img_style):
    # placeholder function for Magenta
    img1 = get_pil_array(img_content)
    img2 = get_pil_array(img_style)
    return img1, img2
