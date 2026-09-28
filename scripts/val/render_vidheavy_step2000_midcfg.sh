#!/bin/bash
# video 主导版(vidheavy) 的 pre-collapse ckpt step_2000 渲染，判 video-dominant 方向是否更自然。
# 与无-video 基线 hallo3_render_factD_lowadv_step11000_midcfg/ 同 CFG 直接 A/B。1步，--no_gt，GPU3。
set -e
export CUDA_VISIBLE_DEVICES=3
export AR_REPO_ROOT=/media/ps/ssd5/ayr/motar
export XNEMO_REPO_ROOT=/media/ps/ssd5/ayr/x-nemo-inference
PY=/home/ayr/miniconda3/envs/xnemo/bin/python
CFG=/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model_depth8.yaml
CKPT=/media/ps/ssd5/ayr/motar/ar_train_output_depth8_diffdepth2_gan_1step_256_factD_vidheavy/20260701_063005_sf_gan_v32/checkpoints/sf_gan_step_2000.pt
TEST=/media/ps/ssd4/ayr/hallo3_test
OUT=/media/ps/ssd4/ayr/hallo3_render_factD_vidheavy_step2000_midcfg

run () {  # $1=tag $2=cfg_audio $3=cfg_text
  echo "==== [$1] cfg_audio=$2 cfg_text=$3 ===="
  $PY $XNEMO_REPO_ROOT/scripts/val/test_ar_model.py \
    --config $CFG --ar_ckpt "$CKPT" --test_dir $TEST \
    --output_dir $OUT/$1 \
    --ar_sampling_steps 1 --num_samples 5 --gpus 0 --no_gt \
    --ar_cfg_audio $2 --ar_cfg_text $3
}

run a1.6_t1.2 1.6 1.2
run a1.7_t1.3 1.7 1.3
run a1.8_t1.4 1.8 1.4
echo "==== VIDHEAVY step2000 MID-CFG DONE ===="
