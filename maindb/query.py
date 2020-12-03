import numpy as np
from PIL import Image

path = "/static/img"


def get_pil_array(img_filename, res=32):
    with Image.open(img_filename) as im:
        img = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)
    return img


def main_predict(img_content, img_style):
    # placeholder function for Magenta
    img1 = get_pil_array(img_content)
    img2 = get_pil_array(img_style)
    return img1, img2
