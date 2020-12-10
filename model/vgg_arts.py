import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

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

paths = tf.placeholder(dtype=tf.string, shape=(None,))
images = tf.map_fn(tf.io.read_file, paths)
images = tf.map_fn(lambda x: tf.image.resize(tf.image.decode_jpeg(x, channels=3), [1024, 1024]), images, dtype=tf.float32)
features = mobilenet(images)[:, 0, 0]


path_in1 = '/mnt/disks/ssd_disk/ac295/image_features/data/Train_Data/'
path_in2 = '/mnt/disks/ssd_disk/ac295/image_features/data/Test_Data/'
path_in1 = '/mnt/disks/ssd_disk/ac295/image_features/data/train/'
path_in2 = '/mnt/disks/ssd_disk/ac295/image_features/data/test/'
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

def get_feature_vector_batch(fs):
    try:
        output = sess.run(features, feed_dict={paths: fs})
        output_dict.update({os.path.basename(f): o for o, f in zip(output, fs)})
    except:
        output = [get_feature_vector(f) for f in fs]
        output_dict.update({os.path.basename(f): o for o, f in zip(output, fs) if o is not None})

c = time.time()
tmp =  get_feature_vector(filenames[5])
print(time.time() - c)

c = time.time()
tmps =  get_feature_vector_batch(filenames[:32])
print(time.time() - c)

output_dict = {}
for i in tqdm(range(32, len(filenames), 32)):
    print(i)
    get_feature_vector_batch(filenames[i-32:i])

get_feature_vector_batch(filenames[i:])

df = pd.DataFrame(output_dict).transpose()
df.to_csv("/mnt/disks/ssd_disk/ac295/image_features/feature_table_full.csv")

# features = [get_feature_vector(f) for f in tqdm(filenames)]
# feature_dict = {os.path.basename(i):j for i, j in zip(filenames, features) if j is not None}
# df = pd.DataFrame(feature_dict).transpose()
# df.to_csv("/mnt/disks/ssd_disk/ac295/image_features/feature_table.csv")
