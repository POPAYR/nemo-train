#!/bin/bash
# SF-MSE 1步 256长度 grounding backbone (best.pt=step7500, val MSE 0.205) 渲染：1步推理，样本1-5。GPU0。
export AR_REPO_ROOT=/media/ps/ssd5/ayr/motar
export XNEMO_REPO_ROOT=/media/ps/ssd5/ayr/x-nemo-inference
/home/ayr/miniconda3/envs/xnemo/bin/python /media/ps/ssd5/ayr/x-nemo-inference/scripts/val/test_ar_model.py \
  --config /media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model_depth8.yaml \
  --ar_ckpt /media/ps/ssd5/ayr/motar/ar_train_output_depth8_diffdepth2_sf_1step_256/20260625_071452_sf_phase1/checkpoints/sf_phase1_best.pt \
  --test_dir /media/ps/ssd4/ayr/hallo3_test \
  --output_dir /media/ps/ssd4/ayr/hallo3_render_sfmse_1step_256 \
  --ar_sampling_steps 1 --num_samples 5 --gpus 0 --no_gt \
  --ar_cfg_audio 4.0 --ar_cfg_text 2.0
