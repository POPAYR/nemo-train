#!/bin/bash
# LCM 4步 (lcm_step_6000) 降 CFG 复渲，只渲样本 2/4，对比原 cfg4/2 浮夸版。GPU0。
set -e
export AR_REPO_ROOT=/media/ps/ssd5/ayr/motar
export XNEMO_REPO_ROOT=/media/ps/ssd5/ayr/x-nemo-inference
PY=/home/ayr/miniconda3/envs/xnemo/bin/python
CFG=/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model_depth8.yaml
LCM=/media/ps/ssd5/ayr/motar/ar_train_output_depth8_diffdepth2_lcm/20260622_095410_lcm/checkpoints/lcm_step_6000.pt
TEST=/media/ps/ssd4/ayr/hallo3_test
OUT=/media/ps/ssd4/ayr/hallo3_render_lcm4s_lowcfg

run () {  # $1=tag $2=cfg_audio $3=cfg_text
  echo "==== LCM4s [$1] cfg_audio=$2 cfg_text=$3  samples 2,4 ===="
  $PY $XNEMO_REPO_ROOT/scripts/val/test_ar_model.py \
    --config $CFG --ar_ckpt $LCM --test_dir $TEST \
    --output_dir $OUT/$1 \
    --ar_sampling_steps 4 --sample_ids 2,4 --gpus 0 --no_gt \
    --ar_cfg_audio $2 --ar_cfg_text $3
}

run a1.0_t1.0 1.0 1.0
run a2.0_t1.5 2.0 1.5
echo "==== LCM4s LOW-CFG DONE ===="
