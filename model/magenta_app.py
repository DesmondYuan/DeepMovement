# Adapted from Magenta console commands

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

        tf.disable_v2_behavior()
        tf.Graph().as_default()
        sess = tf.Session()

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
              content_img_ph, content_image_size)
        else:
          content_img_preprocessed = image_utils.resize_image(
              content_img_ph, content_image_size)

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
        self.stylized_images = stylized_images
        self.content_img_preprocessed = content_img_preprocessed
        self.style_img_preprocessed = style_img_preprocessed
        self.content_img_ph = content_img_ph
        self.bottleneck_feat = bottleneck_feat


    def process_data(self, style_images_paths, content_images_paths):

        # Gets the list of the input images.

        style_img_list = tf.gfile.Glob(style_images_paths)
        content_img_list = tf.gfile.Glob(content_images_paths)

        for content_i, content_img_path in enumerate(content_img_list):
          content_img_np = image_utils.load_np_image_uint8(content_img_path)[:, :, :3]
          content_img_name = os.path.basename(content_img_path)[:-4]

          # Saves preprocessed content image.
          inp_img_croped_resized_np = self.sess.run(
              self.content_img_preprocessed, feed_dict={
                  self.content_img_ph: content_img_np})

          # Computes bottleneck features of the style prediction network for the
          # identity transform.
          identity_params = self.sess.run(
              self.bottleneck_feat, feed_dict={self.style_img_ph: content_img_np})

          for style_i, style_img_path in enumerate(style_img_list):
            style_img_name = os.path.basename(style_img_path)[:-4]
            style_image_np = image_utils.load_np_image_uint8(style_img_path)[:, :, :3]

        self.content_img_np = content_img_np
        self.style_image_np = style_image_np
        self.identity_params = identity_params
        self.style_img_name = style_img_name
        self.content_img_name = content_img_name


    def run(self, output_dir, interpolation_weights):

        style_params = self.sess.run(
            self.bottleneck_feat, feed_dict={self.style_img_ph: self.style_image_np})

        for interp_i, wi in enumerate(interpolation_weights):
          stylized_image_res = self.sess.run(
              self.stylized_images,
              feed_dict={
                  self.bottleneck_feat:
                      self.identity_params * (1 - wi) + style_params * wi,
                  self.content_img_ph:
                      self.content_img_np
              })

          # Saves stylized image.
          image_utils.save_np_image(
              stylized_image_res,
              os.path.join(output_dir, '%s_stylized_%s_%d.jpg' % \
                        (self.content_img_name, self.style_img_name, interp_i)))


magenta_model = Magenta_Model("/mnt/disks/ssd_disk/final/models/",
                 content_square_crop=False, style_square_crop=False,
                 style_image_size=256, content_image_size=256)

magenta_model.process_data(style_images_paths="/mnt/disks/ssd_disk/final/data/content_images/*",
                           content_images_paths="/mnt/disks/ssd_disk/final/data/content_images/*")

magenta_model.run("/mnt/disks/ssd_disk/final/tmp/", [0., 1.])
