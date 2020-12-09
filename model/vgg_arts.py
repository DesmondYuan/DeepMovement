import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

tf.disable_v2_behavior()
mobilenet = hub.Module("https://tfhub.dev/vtab/sup-rotation-100/1")

path = tf.placeholder(dtype=tf.string)
image = tf.io.read_file(path)
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [1024, 1024])
image = tf.expand_dims(image, 0)
feature = mobilenet(image)[0, 0, 0]

path_in1 = '/mnt/disks/ssd_disk/ac295/image_features/data/Train_Data/'
path_in2 = '/mnt/disks/ssd_disk/ac295/image_features/data/Test_Data/'
path_in3 = '/mnt/disks/ssd_disk/ac295/image_features/data/astronomical_image/'

filenames = [path_in1 + f for f in os.listdir(path_in1)] + \
            [path_in2 + f for f in os.listdir(path_in2)] + \
            [path_in3 + f for f in os.listdir(path_in3)]

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

def get_feature_vector(f):
    try:
        return sess.run(feature, feed_dict={path: f})
    except:
        return None

features = [get_feature_vector(f) for f in tqdm(filenames)]
feature_dict = {os.path.basename(i):j for i, j in zip(filenames, features) if j is not None}

df = pd.DataFrame(feature_dict).transpose()
df.to_csv("/mnt/disks/ssd_disk/ac295/image_features/feature_table.csv")
