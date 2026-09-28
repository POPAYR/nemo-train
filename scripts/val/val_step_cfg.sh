python /media/ps/ssd5/ayr/x-nemo-inference/scripts/val/val_step_cfg.py \
  --config /media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml \
  --ar_ckpt /media/ps/ssd5/ayr/motar/ar_video_train_output/exp_20260610_132512/checkpoints/ar_step_24000.pth \
  --test_dir /media/ps/ssd4/ayr/hallo3_test \
  --output_dir /media/ps/ssd4/ayr/compare_samplers_24000step \
  --num_samples 8 --device cuda:3