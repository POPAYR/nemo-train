#!/bin/bash
# 渲染 depth8 LCM-4步（lcm_step_6000）：4 步 LCM 推理。GPU4。--no_gt（GT 用户已另行生成）。
export AR_REPO_ROOT=/media/ps/ssd5/ayr/motar
export XNEMO_REPO_ROOT=/media/ps/ssd5/ayr/x-nemo-inference
python /media/ps/ssd5/ayr/x-nemo-inference/scripts/val/test_ar_model.py \
  --config /media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model_depth8.yaml \
  --ar_ckpt /media/ps/ssd5/ayr/motar/ar_train_output_depth8_diffdepth2_lcm/20260622_095410_lcm/checkpoints/lcm_step_6000.pt \
  --test_dir /media/ps/ssd4/ayr/hallo3_test \
  --output_dir /media/ps/ssd4/ayr/hallo3_render_depth8_lcm4s \
  --ar_sampling_steps 4 --num_samples 4 --gpus 4 --no_gt
