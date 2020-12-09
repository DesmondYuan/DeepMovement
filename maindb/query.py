import numpy as np
from PIL import Image
import hashlib
import json
import os
from magenta.models.arbitrary_image_stylization import arbitrary_image_stylization_build_model as build_model
from magenta.models.image_stylization import image_utils
import tensorflow.compat.v1 as tf
import tf_slim as slim
import pandas as pd


path = "/static/imgs/"
global feature_table
feature_table = pd.read_csv("/static/feature_table.csv", index_col=0)
meta = pd.read_csv("/static/metadata.csv", index_col=0)


def get_pil_array(img_arr, res=32):
    img = np.array(img_arr)
    return img


def magenta_predict(magenta_model, img_content, img_style, weight):
    # placeholder function for Magenta
    img1 = get_pil_array(img_content)
    img2 = get_pil_array(img_style)
    content_images_paths = [save_img(img1)]
    style_images_paths = [save_img(img2)]
    magenta_model.process_data(style_images_paths=style_images_paths,
                               content_images_paths=content_images_paths)

    print("[query.py] The input weight is {} (type: {}).".format(weight, type(weight)))
    if type(weight) is str:
        if weight=='':
            weight = "1.0"
        weight = eval(weight)

    assert 0<=weight<=1

    outfns = magenta_model.run("/static/imgs/", [weight])
    outfns = style_images_paths[0], content_images_paths[0], outfns[0]

    return outfns


def feature_predict(feature_model, img):
    current = feature_model(img)
    fns = list(feature_table.index)
    best_score = 1e10
    best_match = "No match found"
    for fn_iter in fns:
        x = feature_table.loc[fn_iter]
        score = sum((current - x)**2)
        if best_score > score:
            best_match = fn_iter
            best_score = score
    return best_match


def get_metadata(fn):
    record = meta.loc[[fn]].transpose()
    record = record.reset_index()
    return record


def save_img(img):
    img = img.astype(np.uint8)
    outfn = path+md5({'fingerprint': np.diag(img[:,:,0]).tolist()}) + '.jpg'
    im = Image.fromarray(img, mode='RGB')
    im.save(outfn)
    return outfn


def md5(obj):
    key = json.dumps(obj, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()
