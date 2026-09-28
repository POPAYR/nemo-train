#!/usr/bin/env bash
# ==========================================================================
# AR-in-video 微调启动脚本
# ==========================================================================
set -e

# 用到的 GPU（按需修改）
export TOKENIZERS_PARALLELISM=false
# export NCCL_P2P_DISABLE=0
# [改] 不再全局禁用 compile（由 config 的 use_compile 控制；要彻底关掉再把下面这行取消注释）
# export TORCH_COMPILE_DISABLE=1
export AR_REPO_ROOT=/media/ps/ssd5/ayr/motar
export XNEMO_REPO_ROOT=/media/ps/ssd5/ayr/x-nemo-inference
# export OMP_NUM_THREADS=8

CONFIG=/media/ps/ssd5/ayr/x-nemo-inference/configs/train_ar.yaml
GPUS="4,5"
# 单机多卡：--num_gpus 与 CUDA_VISIBLE_DEVICES 卡数一致
deepspeed --master_port 29501 --include localhost:${GPUS} \
 /media/ps/ssd5/ayr/x-nemo-inference/scripts/train/train_ar_xnemo.py --config ${CONFIG}

# 多机示例（取消注释并配置 hostfile）：
# deepspeed --hostfile=hostfile --master_addr=$MASTER_ADDR --master_port=29500 \
#     train_ar_video.py --config ${CONFIG}