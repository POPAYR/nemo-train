#!/bin/bash
export AR_REPO_ROOT=/media/ps/ssd5/ayr/motar
export XNEMO_REPO_ROOT=/media/ps/ssd5/ayr/x-nemo-inference
python /media/ps/ssd5/ayr/x-nemo-inference/scripts/val/test_ar_model.py \
  --config /media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model_depth8.yaml \
  --ar_ckpt /media/ps/ssd5/ayr/motar/ar_train_output_depth8_diffdepth2_gan_4step_fromLCM/20260623_095319_sf_gan_v32/checkpoints/sf_gan_step_9000.pt --test_dir /media/ps/ssd4/ayr/hallo3_test \
  --output_dir /media/ps/ssd4/ayr/hallo3_render_B_step1 \
  --ar_sampling_steps 1 --num_samples 3 --gpus 6 --no_gt \
  --ar_cfg_audio 4.0 --ar_cfg_text 2.0
