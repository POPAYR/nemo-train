#!/bin/bash
# 磁盘清理候选清单 —— 默认 dry-run,只打印不删除。确认后加 --apply 执行。
#   bash cleanup_candidates.sh          # 只看
#   bash cleanup_candidates.sh --apply  # 真删
#
# ★ 分三档:A 完全可删 / B 需确认 / C 建议保留(本脚本不含 C)
XN=/media/ps/ssd5/ayr/x-nemo-inference/output
D=/media/ps/ssd4/ayr/hallo3_frames_512
APPLY=0; [ "$1" = "--apply" ] && APPLY=1

rm_path() {  # $1=路径 $2=说明
  [ -e "$1" ] || return
  local sz=$(du -sh "$1" 2>/dev/null | cut -f1)
  if [ $APPLY -eq 1 ]; then rm -rf "$1"; echo "  已删 $sz  $1  ($2)"
  else echo "  $sz  $1  ($2)"; fi
}

echo "===== A 档:完全可删(已作废的实验产物) ====="
rm_path $XN/s2_val30_500   "30条评测,中途放弃"
rm_path $XN/s2_val30_1000  "同上"
rm_path $XN/s2_val30_1500  "同上"
rm_path $XN/s2_val30_2000  "同上"
rm_path $XN/bbox_A_const   "bbox A/B 验证,结论已记入 FLOW_DISTILL_PROGRESS §5d"
rm_path $XN/bbox_B_real    "同上"
rm_path $XN/s1_ckpt_sweep  "stage1 八个 ckpt 扫描,结论已记入 §5c"
rm_path $XN/s1_old_on_new  "新旧 stage1 控制实验,结论已记入 §5c"
rm_path $D/pose_embed_realbbox "8条验证样本"
rm_path $D/face_mot_feat       "方案3产物,已改用方案1"

echo
echo "===== B 档:需确认(大头) ====="
rm_path $XN/ode_pairs_new  "ODE轨迹231条(未跑完);teacher 要换,这批必然作废"
# flow_stage2_sh3:只保留 step_2500(当前 temporal 暖启动来源),其余可删
for f in $XN/flow_stage2_sh3/stage2_step_*.pt; do
  [ -e "$f" ] || continue
  case "$f" in *step_2500.pt) continue;; esac      # ★ 保留:重训时 temporal 暖启动要用
  rm_path "$f" "旧stage2非必需ckpt"
done
# s1_newdata:只保留 step_1000(当前 stage2 主干来源)
for f in $XN/s1_newdata/stage1_step_*.pt; do
  [ -e "$f" ] || continue
  case "$f" in *step_1000.pt) continue;; esac      # ★ 保留:对照与回滚都要它
  rm_path "$f" "本轮stage1非最优ckpt"
done

echo
echo "===== 保留(不在本脚本内) ====="
echo "  output/flow_stage1_sh3_cfgdrop  2.4G  旧stage1,重训的 --from_ckpt 起点"
echo "  output/s2_newdata                13G  本轮stage2,改 bbox 前后的唯一对照基准"
echo "  output/flow_stage2_sh3/stage2_step_2500.pt   temporal 暖启动来源"
echo "  output/s1_newdata/stage1_step_1000.pt        当前 stage2 主干"
echo "  eval_metrics/testset/hallo3     3.8G  含99个旧残留但占空间极小,不值得冒险"
echo
[ $APPLY -eq 0 ] && echo "(以上为 dry-run。确认无误后:bash cleanup_candidates.sh --apply)"
