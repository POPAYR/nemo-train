#!/bin/bash
# factorized D 版(1步训) step7000 渲染：1步推理，样本1-5(看1235)。GPU5。
export AR_REPO_ROOT=/media/ps/ssd5/ayr/motar
export XNEMO_REPO_ROOT=/media/ps/ssd5/ayr/x-nemo-inference
python /media/ps/ssd5/ayr/x-nemo-inference/scripts/val/test_ar_model.py \
  --config /media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model_depth8.yaml \
  --ar_ckpt /media/ps/ssd5/ayr/motar/ar_train_output_depth8_diffdepth2_gan_1step_256_factD/20260626_205952_sf_gan_v32/checkpoints/sf_gan_step_7000.pt \
  --test_dir /media/ps/ssd4/ayr/hallo3_test \
  --output_dir /media/ps/ssd4/ayr/hallo3_render_factD_1step_step7000 \
  --ar_sampling_steps 1 --num_samples 5 --gpus 5 --no_gt \
  --ar_cfg_audio 4.0 --ar_cfg_text 2.0
