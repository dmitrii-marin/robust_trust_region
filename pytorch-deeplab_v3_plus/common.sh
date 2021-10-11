[ -n "${DATA_ROOT+x}" ] || export DATA_ROOT=/data2

GPU_IDS=0
[ -z "${GPUS}" ] || GPU_IDS=`seq -s, 0 $((${GPUS}-1))`

