# Copyright 2020 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates stylized images with different strengths of a stylization.
For each pair of the content and style images this script computes stylized
images with different strengths of stylization (interpolates between the
identity transform parameters and the style parameters for the style image) and
saves them to the given output_dir.
See run_interpolation_with_identity.sh for example usage.
"""
import os

from magenta.models.arbitrary_image_stylization import arbitrary_image_stylization_build_model as build_model
from magenta.models.image_stylization import image_utils
import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim

class Magenta_Model():

    def __init__(self, checkpoint,
                 content_square_crop=False, style_square_crop=False,
                 style_image_size=256, content_image_size=256):

      with tf.Graph().as_default(), tf.Session() as sess:

        # Defines place holder for the style image.
        self.style_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])

        if style_square_crop:
          style_img_preprocessed = image_utils.center_crop_resize_image(
              style_img_ph, style_image_size)
        else:
          style_img_preprocessed = image_utils.resize_image(self.style_img_ph,
                                                            style_image_size)

        # Defines place holder for the content image.
        content_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
        if content_square_crop:
          content_img_preprocessed = image_utils.center_crop_resize_image(
              content_img_ph, image_size)
        else:
          content_img_preprocessed = image_utils.resize_image(
              content_img_ph, image_size)

        # Defines the model.
        stylized_images, _, _, bottleneck_feat = build_model.build_model(
            content_img_preprocessed,
            style_img_preprocessed,
            trainable=False,
            is_training=False,
            inception_end_point='Mixed_6e',
            style_prediction_bottleneck=100,
            adds_losses=False)

        checkpoint = tf.train.latest_checkpoint(checkpoint)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint, slim.get_variables_to_restore())
        sess.run([tf.local_variables_initializer()])
        init_fn(sess)

      self.sess = sess


    def process_data(self, style_images_paths, content_images_paths):

        # Gets the list of the input style images.
        style_img_list = tf.gfile.Glob(style_images_paths)

        # Gets list of input content images.
        content_img_list = tf.gfile.Glob(content_images_paths)

        for content_i, content_img_path in enumerate(content_img_list):
          content_img_np = image_utils.load_np_image_uint8(content_img_path)[:, :, :3]
          content_img_name = os.path.basename(content_img_path)[:-4]

          # Saves preprocessed content image.
          inp_img_croped_resized_np = self.sess.run(
              content_img_preprocessed, feed_dict={
                  content_img_ph: content_img_np})

          # Computes bottleneck features of the style prediction network for the
          # identity transform.
          identity_params = self.sess.run(
              bottleneck_feat, feed_dict={self.style_img_ph: content_img_np})

          for style_i, style_img_path in enumerate(style_img_list):
            style_img_name = os.path.basename(style_img_path)[:-4]
            style_image_np = image_utils.load_np_image_uint8(style_img_path)[:, :, :3]


    def run(self, output_dir, interpolation_weights, style_image_np):

        style_params = self.sess.run(
            bottleneck_feat, feed_dict={self.style_img_ph: style_image_np})

        for interp_i, wi in enumerate(interpolation_weights):
          stylized_image_res = sess.run(
              stylized_images,
              feed_dict={
                  bottleneck_feat:
                      identity_params * (1 - wi) + style_params * wi,
                  content_img_ph:
                      content_img_np
              })

          # Saves stylized image.
          image_utils.save_np_image(
              stylized_image_res,
              os.path.join(output_dir, '%s_stylized_%s_%d.jpg' % (content_img_name, style_img_name, fn_idx)))
