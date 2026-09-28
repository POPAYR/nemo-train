#!/bin/bash
export AR_REPO_ROOT=/media/ps/ssd5/ayr/motar
export XNEMO_REPO_ROOT=/media/ps/ssd5/ayr/x-nemo-inference
python /media/ps/ssd5/ayr/x-nemo-inference/scripts/val/test_ar_model.py \
  --config /media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model_depth8.yaml \
  --ar_ckpt /media/ps/ssd5/ayr/motar/ar_train_output_depth8_diffdepth2_lcm/20260622_095410_lcm/checkpoints/lcm_step_6000.pt --test_dir /media/ps/ssd4/ayr/hallo3_test \
  --output_dir /media/ps/ssd4/ayr/hallo3_render_fix_lcm4s \
  --ar_sampling_steps 4 --num_samples 3 --gpus 3 --no_gt \
  --ar_cfg_audio 4.0 --ar_cfg_text 2.0
