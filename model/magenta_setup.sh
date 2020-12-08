### To connect to a GPU project
gcloud config set project ac295-data-science-289420
gcloud config set project edv-194519

sudo lsblk
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
sudo mkdir -p ssd_disk
sudo mount -o discard,defaults /dev/sdb /mnt/disks/ssd_disk
sudo chmod a+w /mnt/disks/ssd_disk
cd /mnt/disks/ssd_disk/ac295
sudo apt-get update
sudo apt-get install python3-pip

### Install Magenta
pip3 install --upgrade pip3
sudo apt-get install build-essential libasound2-dev libjack-dev portaudio19-dev
pip3 install magenta
sudo apt-get -y install apt-utils gcc libpq-dev libsndfile-dev

### Prepare ImageNet (~24 hour needed)
# https://sumihui.github.io/tensorflow/2017/10/18/PrepareTheImagenetDataset/
# https://cloud.google.com/tpu/docs/imagenet-setup

# dog subset of ImageNet
mkdir imagenet
cd imagenet
wget http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_train_t3.tar
mkdir ImageNetT3 tf_records
tar -xf ILSVRC2012_img_train_t3.tar -C ImageNetT3
cd ImageNetT3
for D in *.tar; do
  mkdir -p "$(echo $D |cut -d'.' -f 1)" &&
  tar  -xf "$(basename $D)" -C "$(echo $D |cut -d'.' -f 1)"
done
rm -r *.tar

mkdir train validation
cp -r n* train
mv n* validation

# Download ImageNet labels
# wget -O synset_labels.txt https://raw.githubusercontent.com/tensorflow/models/master/research/inception/inception/data/imagenet_2012_validation_synset_labels.txt
wget -O synset_labels.txt http://data.dmlc.ml/mxnet/models/imagenet/synset.txt

# Process imagenet
cd ..
wget https://raw.githubusercontent.com/tensorflow/tpu/master/tools/datasets/imagenet_to_gcs.py
pip3 install google-cloud-storage
python3 imagenet_to_gcs.py \
      --nogcs_upload \
      --raw_data_dir=ImageNetT3 \
      --local_scratch_dir=./tf_records
