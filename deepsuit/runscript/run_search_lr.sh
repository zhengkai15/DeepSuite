#!/bin/bash
set -e # Exit on error
set -x # Print commands

# def some common info: path
COMMON="" # path common 
FLAG="" # path flag

# Python interpreter path
PY=${COMMON}/miniconda3/envs/FuXi-Sparse/bin/python

# Working directory
WORK_DIR=${COMMON}/${FLAG}
cd "$WORK_DIR"

# config
CONFIG=${COMMON}/${FLAG}/conf/config.yaml
MODE=train
# Experiment directory
EXP_DIR_ROOT=${COMMON}/${FLAG}/aexp
TRAIN_BS=1
WEIGHT_DECAY=0.2

# Array of lr values to test
LRS=(0.01 0.001 0.0005 0.0001)
LRS=(0.001 0.0005)
EXP_NAME="lr_search_WEIGHT_DECAY_0.2_USE_NOISE_DROPOUT_0.5"

# Array of lr values to test
USE_NOISE=true
NOISE_LEVELS=0.001
MODEL_NAME=temporal_model
DROPOUT=0.5

PIDS=()
GPU_IDS=(0 1) # Available GPUs
# GPU_IDS=(0) # Available GPUs

# Function to get next available GPU
get_next_gpu() {
    local current=$1
    local next=$(( (current + 1) % ${#GPU_IDS[@]} ))
    echo ${GPU_IDS[$next]}
}

# Function to check process status
check_process() {
    local pid=$1
    local exp_dir=$2
    if ps -p $pid -o pid,cmd > /dev/null; then
        echo "Training process $pid is running."
        # echo "$EXP_NAME || $(date +%Y%m%d_%H%M%S) || bash $0 $@ || $pid" >> "$EXP_DIR_ROOT/EXP.log"
        echo "$EXP_NAME || $(date +%Y%m%d_%H%M%S) || $pid" >> "$EXP_DIR_ROOT/EXP.log"
    else
        echo "Error: Training process $pid is not running."
        return 1
    fi
}

# Launch training jobs
current_gpu=${GPU_IDS[0]}
for item in "${LRS[@]}"; do
    export CUDA_VISIBLE_DEVICES=$current_gpu
    
    exp_dir="$EXP_DIR_ROOT/$EXP_NAME/lr_$item"
    mkdir -p "$exp_dir"
    
    # Launch training process
    nohup ${PY} -u train_main.py \
        --noise.use_noise "${USE_NOISE}" \
        --noise.noise_level "${NOISE_LEVELS}" \
        --model.name "${MODEL_NAME}" \
        --model.dropout "${DROPOUT}" \
        --mode "${MODE}" \
        --config "${CONFIG}" \
        --lr.value "${item}" \
        --optim.adamw.weight_decay "${WEIGHT_DECAY}" \
        --exp.dir "${exp_dir}" \
        --training.batch_size "${TRAIN_BS}" \
        > "${exp_dir}/train_${item}.log" 2>&1 &
    
    pid=$!
    echo $pid > "$exp_dir/train.pid"
    PIDS+=($pid)
    
    # Switch to next GPU
    current_gpu=$(get_next_gpu $current_gpu)
    
    # Check process status after brief delay
    sleep 20
    check_process $pid "$exp_dir" || continue
    # break
done

# Wait for all processes to complete
echo "Waiting for processes: ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
    echo "Waiting for process $pid..."
    if wait $pid; then
        touch "$EXP_DIR_ROOT/$EXP_NAME/${pid}.success"
    else
        echo "Process $pid failed with status $?"
        touch "$EXP_DIR_ROOT/$EXP_NAME/${pid}.failed"
    fi
done

echo "All training jobs completed successfully."