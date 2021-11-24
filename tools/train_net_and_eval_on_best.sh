CONFIG_FILE=$1
PORT=$2
if [ -n "$1" ]; then
    echo "config file: $CONFIG_FILE"
else
    echo "no config file specified, exit..."
    exit
fi

if [ -n "$2" ]; then
    echo "using tcp address: tcp://127.0.0.1:"$PORT
    DIST_URL="tcp://127.0.0.1:"$PORT

else
    echo "no port specified, use default 50257..."
    DIST_URL="tcp://127.0.0.1:50257"
fi

# train
echo "start to train..."
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python tools/train_net.py \
--config-file $CONFIG_FILE \
--num-gpus 4 \
--dist-url $DIST_URL

# construct the best model path
log_dir=${CONFIG_FILE/configs/logs}
log_dir=${log_dir/.yml/}
best_model_dir=$log_dir/model_best.pth

# test
echo "start to eval on best model..."
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python tools/train_net.py \
--config-file $CONFIG_FILE \
--num-gpus 4 \
--dist-url $DIST_URL \
--eval-only \
MODEL.WEIGHTS $best_model_dir
