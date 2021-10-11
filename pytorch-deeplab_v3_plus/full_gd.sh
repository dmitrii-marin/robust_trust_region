set -ex

source common.sh

suffix=0.3

CUDA_VISIBLE_DEVICES=0 python3 train_with_dcr.py \
  --backbone mobilenet --lr 0.0000007 --workers 1 --epochs 60 \
  --batch-size 12 \
  --checkname mn-gd-linear-crf  --dataset pascal --train_dataset_suffix "$suffix" \
  --eval-interval 1 --save-interval 0 \
  --resume checkpoint_epoch_60.pth.tar --ft \
  --potts-weight 100 \
  --use-linear-relaxation \
  # --train-shuffle 0 \
  # --no-aug \
  # --freeze-bn True  \
  # --no-val \
  # --single-image-training -9 \
  # --use-dcr \
