!#/bin/bash

sudo mkfs.ext4 -F /dev/sdb
sudo mkdir -p /mnt/disks/ssd
sudo mount /dev/sdb /mnt/disks/ssd
sudo chmod a+w /mnt/disks/ssd

gsutil -m cp -r gs://traininus/val.lmdb /mnt/disks/ssd
gsutil -m cp -r gs://traininus/train.lmdb /mnt/disks/ssd
