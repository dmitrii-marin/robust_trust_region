set -xe

source common.sh
source backbone.sh

[ -n "${suffix+x}" ] || suffix=0.3
[ -n "$ERROR_PROB" ] || ERROR_PROB=0.95160
[ -n "$LR" ] || LR=0.0007
[ -n "$TR_WEIGHT" ] || TR_WEIGHT=0.1
[ -n "$HIDDEN_UPDATE" ] || HIDDEN_UPDATE=5
[ -n "$SCRATCH" ] || EXTRA_ARGS="$EXTRA_ARGS --resume ${CP_PREFIX}_checkpoint${suffix}_epoch_*.pth.tar --ft"
[ -n "$VIZ" ] || VIZ=0
[ -n "$SAVE" ] || SAVE=0
[ -n "$EPOCHS" ] || EPOCHS=60


python3 train_with_dcr.py \
  --gpu-ids $GPU_IDS \
  --backbone $BACKBONE --workers 44 --epochs 60 \
  --batch-size 12 \
  --checkname ${CP_PREFIX}-tr-g${TR_WEIGHT}-error${ERROR_PROB}  --dataset pascal --train_dataset_suffix "$suffix" \
  --eval-interval 5 --save-interval $SAVE \
  --potts-weight 100 \
  --use-dcr AlphaExpansion \
  --gc-scale 1 \
  --tr-weight $TR_WEIGHT \
  --hidden-update $HIDDEN_UPDATE \
  --loss ce \
  --lr $LR \
  --use-pce-at-tr 0 \
  --tr-restricted \
  --tr-error-prob $ERROR_PROB \
  --tr-error-model Const \
  --viz-images-per-epoch $VIZ \
  $EXTRA_ARGS \
  # --proposals $DATA_ROOT/alpha_proposals \
  # --train-shuffle 0 \
  # --no-aug \
  # --no-val \
  # --single-image-training -2 \
  # --freeze-bn True   \

