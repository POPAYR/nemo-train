python /media/ps/ssd5/ayr/x-nemo-inference/scripts/val/test_ar_model.py \
  --config /media/ps/ssd5/ayr/x-nemo-inference/configs/test_ar_model.yaml \
  --ar_ckpt /media/ps/ssd5/ayr/motar/ar_train_output_depth12_diffdepth4_sf1/20260613_104112_sf_phase1/checkpoints/sf_phase1_step_6000.pt \
  --test_dir /media/ps/ssd4/ayr/hallo3_test \
  --output_dir /media/ps/ssd4/ayr/hallo3_ar_w_xnemo_output_12depth_4diff_sf_6k \
  --ar_sampling_steps 1 --num_samples 9 --gpus 6 --no_gt