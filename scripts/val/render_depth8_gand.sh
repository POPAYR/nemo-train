#!/bin/bash
# 渲染 depth8 GAN-D（再平衡 step_4000）：4 步 LCM 推理。GPU5。--no_gt（GT 已在 lcm4s 目录）。
export AR_REPO_ROOT=/media/ps/ssd5/ayr/motar
export XNEMO_REPO_ROOT=/media/ps/ssd5/ayr/x-nemo-inference
python /media/ps/ssd5/ayr/x-nemo-inference/scripts/val/test_ar_model.py \
  --config /media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model_depth8.yaml \
  --ar_ckpt /media/ps/ssd5/ayr/motar/ar_train_output_depth8_diffdepth2_gan_4step_rebal/20260623_155150_sf_gan_v32/checkpoints/sf_gan_step_4000.pt \
  --test_dir /media/ps/ssd4/ayr/hallo3_test \
  --output_dir /media/ps/ssd4/ayr/hallo3_render_depth8_gand \
  --ar_sampling_steps 4 --num_samples 4 --gpus 5 --no_gt
