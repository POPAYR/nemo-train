#!/bin/bash
# stage1 shift 微调的选点 watcher:每出一个新 ckpt,并行跑两个信号。
#
#  ① σ 分辨 v-MSE 诊断(GPU_A):对照起点 stage1_step_3000,看高噪段补上多少、低噪段退多少
#  ② 合成渲染评测(GPU_B):把新 stage1 空间主干合进 CUM2000 的 temporal 再渲染
#     ★ 因为 CUM2000 的空间键 ≡ stage1_step_3000(stage2 冻结主干,逐位相同),
#       所以基线就是已测好的 CUM2000(FVD 260.0 / FID 43.6 / FLK 0.908),唯一变量=stage1 空间权重
#     口径与 D 组一致:35步 / 均匀网格 / cfg2.5 / window0 / hallo3 subset30
#
# 用法: bash scripts/val/watch_stage1.sh 6 7
set -u
XN=/media/ps/ssd5/ayr/x-nemo-inference
BENCH=/media/ps/ssd5/ayr/eval_metrics/decoder_bench
CKPTDIR=$XN/output/flow_stage1_shift3
BASE=$XN/output/flow_stage1/stage1_step_3000.pt
FACE=/home/ayr/miniconda3/envs/face/bin/python
GA=${1:-6}; GB=${2:-7}
mkdir -p $XN/output/diag_stage1_shift3
cd $XN
while true; do
  for f in $(ls $CKPTDIR/stage1_step_*.pt 2>/dev/null | sort -t_ -k3 -n); do
    n=$(basename $f .pt); n=${n#stage1_step_}
    DONE=$XN/output/diag_stage1_shift3/step_$n.json
    [ -f "$DONE" ] && continue
    s1=$(stat -c%s $f); sleep 20; s2=$(stat -c%s $f)
    [ "$s1" != "$s2" ] && continue          # 还在写,下轮再来
    echo "[$(date +%H:%M)] ===== stage1_step_$n ====="

    # ① σ 诊断
    ( CUDA_VISIBLE_DEVICES=$GA python -u scripts/val/diag_sigma_error.py --gpu 0 --clips 12 --no_temporal \
        --ckpts $BASE $f --out $DONE > $XN/output/diag_stage1_shift3/step_$n.log 2>&1
      echo "[$(date +%H:%M)]   ① σ诊断完成 step_$n" ) &
    PA=$!

    # ② 合成 + 渲染 + 评测
    ( V=abl_s1sh3_$n
      TMP=$XN/output/_composed_$n.pt
      python -u scripts/val/compose_stage1_ckpt.py --stage1 $f --out $TMP \
        > $XN/output/diag_stage1_shift3/compose_$n.log 2>&1 || { echo "  ✗ compose 失败"; exit 1; }
      cd $BENCH
      python - "$V" "$TMP" "$n" <<'PY'
import json,sys,collections
v,ck,n=sys.argv[1],sys.argv[2],sys.argv[3]
p="configs/variants.json"; V=json.load(open(p),object_pairs_hook=collections.OrderedDict)
V[v]=collections.OrderedDict(label=f"stage1 shift3 @{n}步 (+CUM2000 temporal)",family="bidir",
  objective="flow",ckpt=ck,ckpt_key="denoising_unet",
  sampler=collections.OrderedDict(steps=35,cfg=2.5,window=0,shift=1.0),
  note="stage1 选点:空间主干=新 ckpt, temporal=CUM2000。基线 CUM2000(其空间键≡stage1_step_3000)。"
       "口径与 D 组一致(35步/均匀/cfg2.5/window0),唯一变量=stage1 空间权重")
json.dump(V,open(p,"w"),ensure_ascii=False,indent=2)
PY
      python -u gen_variants.py --variant $V --gpu $GB --datasets hallo3 --subset subset30 \
        > logs/gen_$V.log 2>&1
      if grep -q "^DONE" logs/gen_$V.log; then
        CUDA_VISIBLE_DEVICES=$GB $FACE -u eval_variants.py --variants $V --datasets hallo3 \
          --metrics paired fid fvd dyn txf > logs/eval_$V.log 2>&1
        CUDA_VISIBLE_DEVICES=$GB $FACE -u txf_hfe.py --ds hallo3 --size 512 --variants $V \
          > logs/hfe_$V.log 2>&1
        echo "[$(date +%H:%M)]   ② 渲染评测完成 $V"
      else
        echo "[$(date +%H:%M)]   ✗ 渲染失败 $V"; tail -3 logs/gen_$V.log
      fi
      rm -f $TMP ) &                        # 合成 ckpt 3.4G,用完即删(磁盘只剩 100G)
    PB=$!
    wait $PA $PB
    echo "[$(date +%H:%M)] ===== step_$n 两项均完成 ====="
  done
  # 训练结束且无新 ckpt 则退出
  pgrep -f "[f]low_stage1_image.py" > /dev/null || { echo "[$(date +%H:%M)] 训练已结束,watcher 退出"; break; }
  sleep 300
done
