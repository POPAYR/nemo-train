#!/bin/bash
# CFG sweep 验证假设：渲染浮夸是否源于推理 CFG 与"训练即 cfg=1"失配。
# 同一 GAN 候选 (factD lowadv step10000) 在多档 CFG，各渲 2 样本；外加 SF-MSE@cfg1 对照。GPU0。
set -e
export AR_REPO_ROOT=/media/ps/ssd5/ayr/motar
export XNEMO_REPO_ROOT=/media/ps/ssd5/ayr/x-nemo-inference
PY=/home/ayr/miniconda3/envs/xnemo/bin/python
CFG=/media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model_depth8.yaml
GAN=/media/ps/ssd5/ayr/motar/ar_train_output_depth8_diffdepth2_gan_1step_256_factD_lowadv/20260628_070514_sf_gan_v32/checkpoints/sf_gan_step_10000.pt
SFMSE=/media/ps/ssd5/ayr/motar/ar_train_output_depth8_diffdepth2_sf_1step_256/20260625_071452_sf_phase1/checkpoints/sf_phase1_best.pt
TEST=/media/ps/ssd4/ayr/hallo3_test
OUT=/media/ps/ssd4/ayr/hallo3_render_cfg_sweep

run () {  # $1=ckpt $2=tag $3=cfg_audio $4=cfg_text
  echo "==== [$2] cfg_audio=$3 cfg_text=$4 ===="
  $PY $XNEMO_REPO_ROOT/scripts/val/test_ar_model.py \
    --config $CFG --ar_ckpt "$1" --test_dir $TEST \
    --output_dir $OUT/$2 \
    --ar_sampling_steps 1 --num_samples 2 --gpus 0 --no_gt \
    --ar_cfg_audio $3 --ar_cfg_text $4
}

run "$GAN"   gan_a1.0_t1.0  1.0 1.0
run "$GAN"   gan_a1.5_t1.0  1.5 1.0
run "$GAN"   gan_a2.0_t1.5  2.0 1.5
run "$SFMSE" sfmse_a1.0_t1.0 1.0 1.0
echo "==== CFG SWEEP DONE ===="
