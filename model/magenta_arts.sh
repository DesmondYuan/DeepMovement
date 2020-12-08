# Art images
sudo gsutil -o GSUtil:parallel_composite_upload_threshold=150M cp gs://gent2/ac295/DeepMovement/*.zip data
sudo apt-get install unzip
unzip content_images.zip
unzip gap_images.zip
unzip astronomy_low_res.zip
mkdir style_images
cp gap_images/*[0-1].jpg style_images # total 9k images + 100 images - picking random 20% of 9k + 100 = 2k in total
cp astronomy_low_res/* style_images
rm -r __MACOSX astronomy_low_res
cd /mnt/disks/ssd_disk/ac295/data/style_images
rm gap_18101.jpg gap_14761.jpg

ls -l /mnt/disks/ssd_disk/ac295/data/style_images |wc -l  # 1971

# Make TFRecord (small dataset 100 images)
path="/mnt/disks/ssd_disk/ac295"
STYLE_IMAGES_PATHS="$path"/data/style_images/*
RECORDIO_PATH="$path"/data/style_images.tfrecords

image_stylization_create_dataset \
    --style_files=$STYLE_IMAGES_PATHS \
    --output_file=$RECORDIO_PATH \
    --compute_gram_matrices=False \
    --logtostderr


# Training on arts datase
CUDA_VISIBLE_DEVICES=0 arbitrary_image_stylization_train \
      --batch_size=32 \
      --imagenet_data_dir=imagenet/tf_records/train \
      --vgg_checkpoint=pretrained_models/vgg_16.ckpt  \
      --inception_v3_checkpoint=pretrained_models/inception_v3.ckpt \
      --style_dataset_file=$RECORDIO_PATH \
      --train_dir=output_arts \
      --content_weights={\"vgg_16/conv3\":2.0} \
      --random_style_image_size=False \
      --augment_style_images=False \
      --center_crop=True \
      --save_interval_secs=600 \
      --save_summaries_secs=600 \
      --logtostderr \
      --verbosity=1 \
      --train_steps=60000
# iter 7200: 900k -> 150k
# iter 14400: -> 110k
# iter 24000: -> 90k
# iter 28000: -> 80k
# iter 60000: -> 70k


# Predict
rm data/style_images_test/*
cp data/gap_images/gap_11218.jpg data/style_images_test
cp data/gap_images/gap_11410.jpg data/style_images_test
cp data/gap_images/gap_11242.jpg data/style_images_test
cp data/gap_images/gap_11426.jpg data/style_images_test
cp data/gap_images/gap_11739.jpg data/style_images_test
cp data/gap_images/gap_14785.jpg data/style_images_test
cp data/gap_images/gap_14883.jpg data/style_images_test
cp data/gap_images/gap_15000.jpg data/style_images_test
cp data/gap_images/gap_15055.jpg data/style_images_test
cp data/gap_images/gap_15067.jpg data/style_images_test
cp data/gap_images/gap_16788.jpg data/style_images_test
cp data/gap_images/gap_20301.jpg data/style_images_test
cp data/gap_images/gap_16788.jpg data/style_images_test
cp data/gap_images/gap_19856.jpg data/style_images_test
cp data/gap_images/gap_19830.jpg data/style_images_test
cp data/style_images/a0110.jpg data/style_images_test
cp data/style_images/a0111.jpg data/style_images_test
cp data/style_images/a070.jpg data/style_images_test
cp data/style_images/a02.jpg data/style_images_test
cp data/style_images/a03.jpg data/style_images_test
cp data/style_images/a08.jpg data/style_images_test
cp data/style_images/a012.jpg data/style_images_test
cp data/style_images/a056.jpg data/style_images_test
cp data/style_images/a048.jpg data/style_images_test


INTERPOLATION_WEIGHTS='[0.5,1.0]'
arbitrary_image_stylization_with_weights \
  --checkpoint=output_arts/model.ckpt-60000 \
  --output_dir=figures_arts_using_public_model \
  --style_images_paths=/mnt/disks/ssd_disk/ac295/data/style_images_test/* \
  --content_images_paths=/mnt/disks/ssd_disk/ac295/data/content_images/* \
  --image_size=256 \
  --content_square_crop=False \
  --style_image_size=256 \
  --style_square_crop=False \
  --interpolation_weights=$INTERPOLATION_WEIGHTS \
  --logtostderr

tar -czf figures_arts.tar figures_arts
sudo gsutil cp -r figures_arts.tar gs://gent2/ac295/

tar -czf style_images_test.tar data/style_images_test
sudo gsutil cp -r style_images_test.tar gs://gent2/ac295/
tar -czf figures_arts_using_public_model.tar figures_arts_using_public_model
sudo gsutil cp -r figures_arts_using_public_model.tar gs://gent2/ac295/
