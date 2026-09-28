#!/bin/bash
# v3 step_3000 (低噪采样+w_vid2+GAN归位) 渲染判 jitter 是否消。1步,--no_gt,GPU0,cfg1.7/1.3。对比 v2(抖)+无video基线。
set -e
export CUDA_VISIBLE_DEVICES=0
export AR_REPO_ROOT=/media/ps/ssd5/ayr/motar
export XNEMO_REPO_ROOT=/media/ps/ssd5/ayr/x-nemo-inference
/home/ayr/miniconda3/envs/xnemo/bin/python /media/ps/ssd5/ayr/x-nemo-inference/scripts/val/test_ar_model.py \
  --config /media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model_depth8.yaml \
  --ar_ckpt "/media/ps/ssd5/ayr/motar/ar_train_output_depth8_diffdepth2_gan_1step_256_factD_video_v3/20260702_080807_sf_gan_v32/checkpoints/sf_gan_step_3000.pt" \
  --test_dir /media/ps/ssd4/ayr/hallo3_test \
  --output_dir /media/ps/ssd4/ayr/hallo3_render_v3_step3000_a1.7 \
  --ar_sampling_steps 1 --num_samples 5 --gpus 0 --no_gt \
  --ar_cfg_audio 1.7 --ar_cfg_text 1.3
echo "==== v3 step3000 render DONE ===="
