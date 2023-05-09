unzip -q /content/drive/MyDrive/data/DIV2K/DIV2K_train_HR.zip -d /content/BasicSR/datasets
echo 'unzip DIV2K_train_HR done'
unzip -q /content/drive/MyDrive/data/DIV2K/DIV2K_valid_HR.zip -d /content/BasicSR/datasets
echo 'unzip DIV2K_valid_HR done'
unzip -q /content/drive/MyDrive/data/Flickr2K/Flickr2K.zip -d /content/BasicSR/datasets
echo 'unzip Flickr2K done'
unzip -q /content/drive/MyDrive/data/Classical/Set5.zip -d /content/BasicSR/datasets
echo 'unzip Set5 done'
python /content/drive/MyDrive/data/predata.py -w --scale LRbicx4 --hr_img_dir /content/BasicSR/datasets/DIV2K_train_HR --lr_img_dir /content/BasicSR/datasets/DF2K
python /content/drive/MyDrive/data/predata.py -w --scale LRbicx4 --hr_img_dir /content/BasicSR/datasets/DIV2K_valid_HR --lr_img_dir /content/BasicSR/datasets/DF2K_val
python /content/drive/MyDrive/data/predata.py -w --scale LRbicx4 --hr_img_dir /content/BasicSR/datasets/Flickr2K --lr_img_dir /content/BasicSR/datasets/DF2K
cd /content/BasicSR/ && python basicsr/utils/utils_blindsr.py