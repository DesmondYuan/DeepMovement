import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

class Feature_Model():

    def __init__(self):

        tf.disable_v2_behavior()
        mobilenet = hub.Module("https://tfhub.dev/vtab/sup-rotation-100/1")
        path = tf.placeholder(dtype=tf.string)
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [1024, 1024])
        image = tf.expand_dims(image, 0)
        feature = mobilenet(image)[0, 0, 0]
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        self.sess = sess
        self.feature = feature
        self.path_in = path

    def __call__(self, filename):

        sess = self.sess
        path = self.path_in
        feature = self.feature
        res = sess.run(feature, feed_dict={path: filename})
        return res
