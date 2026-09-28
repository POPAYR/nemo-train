#!/bin/bash
export AR_REPO_ROOT=/media/ps/ssd5/ayr/motar
export XNEMO_REPO_ROOT=/media/ps/ssd5/ayr/x-nemo-inference
python /media/ps/ssd5/ayr/x-nemo-inference/scripts/val/test_ar_model.py \
  --config /media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml \
  --ar_ckpt /media/ps/ssd5/ayr/motar/ar_train_output_depth12_diffdepth4_gan_video/20260622_083717_sf_gan_v32/checkpoints/sf_gan_step_15000.pt \
  --test_dir /media/ps/ssd4/ayr/hallo3_test \
  --output_dir /media/ps/ssd4/ayr/hallo3_render_depth12_video \
  --ar_sampling_steps 1 --num_samples 5 --gpus 5 --no_gt
