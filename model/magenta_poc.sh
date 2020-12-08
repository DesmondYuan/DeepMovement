# Make TFRecord (small dataset 100 images)
path=$(pwd)
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xvzf dtd-r1.0.1.tar.gz
STYLE_IMAGES_PATHS="$path"/dtd/images/cobwebbed/*.jpg
RECORDIO_PATH="$path"/dtd_cobwebbed.tfrecord

image_stylization_create_dataset \
    --style_files=$STYLE_IMAGES_PATHS \
    --output_file=$RECORDIO_PATH \
    --compute_gram_matrices=False \
    --logtostderr



# Prepare VGG and InceptionV3
# https://github.com/tensorflow/models/tree/master/research/slim
mkdir ../pretrained_models
cd ../pretrained_models
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xf vgg_16_2016_08_28.tar.gz

wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xf inception_v3_2016_08_28.tar.gz
rm *.gz
cd ..



# Training (using 1 GPU: 1 item/s; using 8 CPUs: 56 s/item)
logdir="$path"/dtd_cobwebbed.log

CUDA_VISIBLE_DEVICES=0 arbitrary_image_stylization_train \
      --batch_size=32 \
      --imagenet_data_dir=imagenet/tf_records/train \
      --vgg_checkpoint=pretrained_models/vgg_16.ckpt  \
      --inception_v3_checkpoint=pretrained_models/inception_v3.ckpt \
      --style_dataset_file=$RECORDIO_PATH \
      --train_dir="$logdir"/train_dir \
      --content_weights={\"vgg_16/conv3\":2.0} \
      --random_style_image_size=False \
      --augment_style_images=False \
      --center_crop=True \
      --save_interval_secs=600 \
      --save_summaries_secs=600 \
      --logtostderr \
      --verbosity=1 \
      --train_steps=100

CUDA_VISIBLE_DEVICES='' arbitrary_image_stylization_train \
      --batch_size=8 \
      --imagenet_data_dir=imagenet/tf_records/train \
      --vgg_checkpoint=pretrained_models/vgg_16.ckpt  \
      --inception_v3_checkpoint=pretrained_models/inception_v3.ckpt \
      --style_dataset_file=$RECORDIO_PATH \
      --train_dir="$logdir"/train_dir \
      --content_weights={\"vgg_16/conv3\":2.0} \
      --random_style_image_size=False \
      --augment_style_images=False \
      --center_crop=True \
      --save_interval_secs=600 \
      --save_summaries_secs=600 \
      --logtostderr \
      --verbosity=1 \
      --train_steps=100



# Download pretrained Magenta model (not used)
wget https://storage.googleapis.com/download.magenta.tensorflow.org/models/arbitrary_style_transfer.tar.gz
tar -xf arbitrary_style_transfer.tar.gz
rm *.gz

cd ..
mkdir output
cp pretrained_models/arbitrary_style_transfer/model.ckpt.index output/model.ckpt-99999.index
cp pretrained_models/arbitrary_style_transfer/model.ckpt.data-00000-of-00001 output/model.ckpt-99999.data-00000-of-00001
cp pretrained_models/arbitrary_style_transfer/model.ckpt.meta output/model.ckpt-99999.meta
echo 'model_checkpoint_path: "model.ckpt-9999"' >> output/checkpoint
echo 'all_model_checkpoint_paths: "model.ckpt-9999"' >> output/checkpoint



# Training POC on cobwebbed datase
CUDA_VISIBLE_DEVICES=0 arbitrary_image_stylization_train \
      --batch_size=32 \
      --imagenet_data_dir=imagenet/tf_records/train \
      --vgg_checkpoint=pretrained_models/vgg_16.ckpt  \
      --inception_v3_checkpoint=pretrained_models/inception_v3.ckpt \
      --style_dataset_file=$RECORDIO_PATH \
      --train_dir=output_dtd \
      --content_weights={\"vgg_16/conv3\":2.0} \
      --random_style_image_size=False \
      --augment_style_images=False \
      --center_crop=True \
      --save_interval_secs=600 \
      --save_summaries_secs=600 \
      --logtostderr \
      --verbosity=1 \
      --train_steps=7200
# 3Million -> 200k




# Load weights (not used)
# import tensorflow as tf
# tf.train.import_meta_graph("pretrained_models/arbitrary_style_transfer/model.ckpt.meta")
# for n in tf.get_default_graph().as_graph_def().node:
#    print(n)
#
# DEFAULT_CONTENT_WEIGHTS = '{"vgg_16/conv3": 1}'
# DEFAULT_STYLE_WEIGHTS = ('{"vgg_16/conv1": 0.5e-3, "vgg_16/conv2": 0.5e-3,'
#                          ' "vgg_16/conv3": 0.5e-3, "vgg_16/conv4": 0.5e-3}')


# Change the source codes for larger num_workers (not used)
# cd ~/.conda/envs/conda37/lib/python3.7/site-packages/magenta
# vi models/image_stylization/image_utils.py
# vi models/arbitrary_image_stylization/arbitrary_image_stylization_train.py  #line 78


# Predict
INTERPOLATION_WEIGHTS='[0.0,0.2,0.4,0.6,0.8,1.0]'
arbitrary_image_stylization_with_weights \
  --checkpoint=output_dtd/model.ckpt-7200 \
  --output_dir=figures \
  --style_images_paths="$path"/dtd/images/cobwebbed/cobwebbed_004*.jpg \
  --content_images_paths=/mnt/disks/ssd_disk/ac295/imagenet/ImageNetT3/train/n02085620/n02085620_34*.JPEG   \
  --image_size=256 \
  --content_square_crop=False \
  --style_image_size=256 \
  --style_square_crop=False \
  --interpolation_weights=$INTERPOLATION_WEIGHTS \
  --logtostderr

tar -czf figures.tar figures
sudo gsutil cp -r figures.tar gs://gent2/ac295/


# Change init_fn to load pretrained models (not used)
# cd ~/.conda/envs/conda37/lib/python3.7/site-packages/magenta
# vi models/arbitrary_image_stylization/arbitrary_image_stylization_train.py  #line 65 and 146
#
# CUDA_VISIBLE_DEVICES=0 arbitrary_image_stylization_train \
#       --batch_size=32 \
#       --checkpoint=pretrained_models/arbitrary_style_transfer/ \
#       --imagenet_data_dir=imagenet/tf_records/train \
#       --vgg_checkpoint=pretrained_models/vgg_16.ckpt  \
#       --inception_v3_checkpoint=pretrained_models/inception_v3.ckpt \
#       --style_dataset_file=$RECORDIO_PATH \
#       --train_dir=output_dtd_w_pretraining \
#       --content_weights={\"vgg_16/conv3\":2.0} \
#       --random_style_image_size=False \
#       --augment_style_images=False \
#       --center_crop=True \
#       --save_interval_secs=600 \
#       --save_summaries_secs=600 \
#       --logtostderr \
#       --verbosity=1 \
#       --train_steps=20
