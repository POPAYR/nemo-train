#!/bin/bash
# vidheavy_v2 step_5000 (w_vid full 4.5, std_time~0.82 强 motion) 渲染判自然 vs 抖动。1步,--no_gt,GPU3,cfg1.7/1.3。
set -e
export CUDA_VISIBLE_DEVICES=3
export AR_REPO_ROOT=/media/ps/ssd5/ayr/motar
export XNEMO_REPO_ROOT=/media/ps/ssd5/ayr/x-nemo-inference
/home/ayr/miniconda3/envs/xnemo/bin/python /media/ps/ssd5/ayr/x-nemo-inference/scripts/val/test_ar_model.py \
  --config /media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model_depth8.yaml \
  --ar_ckpt "ar_train_output_depth8_diffdepth2_gan_1step_256_factD_vidheavy_v2/20260701_145120_sf_gan_v32/checkpoints/sf_gan_step_5000.pt" \
  --test_dir /media/ps/ssd4/ayr/hallo3_test \
  --output_dir /media/ps/ssd4/ayr/hallo3_render_vidheavy_v2_step5000_a1.7 \
  --ar_sampling_steps 1 --num_samples 5 --gpus 0 --no_gt \
  --ar_cfg_audio 1.7 --ar_cfg_text 1.3
echo "==== vidheavy_v2 step5000 render DONE ===="
