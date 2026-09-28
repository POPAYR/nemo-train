#!/bin/bash
# 实验机自检:路径、关键文件、Python 依赖、GPU、磁盘。首次部署和每次换数据后跑一次。
source "$(dirname "$0")/env.sh" || exit 1
ok=0; bad=0
chk(){ if [ -e "$1" ]; then echo "  ✓ $2"; ok=$((ok+1)); else echo "  ✗ $2  ($1 不存在)"; bad=$((bad+1)); fi; }
echo "== 路径变量"; $XN_PY -m src.utils.paths
echo "== 原始数据(搬来的)"
chk "$XN_HALLO3_RAW"                     "原始 mp4 目录"
chk "$XN_HALLO3/manifest.txt"            "定稿清单 manifest.txt"
chk "$XN_HALLO3/emo_pose_caption"        "caption"
echo "== 处理产物(remote/prepare_data.sh 生成)"
chk "$XN_HALLO3/frame_latent"            "hallo3 frame_latent"
chk "$XN_HALLO3/pose_embed_real"         "hallo3 pose_embed_real"
chk "$XN_HALLO3/audio_wav"               "hallo3 audio_wav(渲染合音频)"
chk "$XN_HALLO3/train_data_ge64.txt"     "hallo3 训练清单 train_data_ge64.txt"
chk "$XN_TESTSET/manifest.json"          "测试集 manifest.json"
chk "$XN_TESTSET/hallo3_subset30.json"   "测试集 hallo3_subset30.json"
chk "$XN_FVD_I3D"                        "FVD I3D 权重"
echo "== 预训练权重"
for d in sd-image-variations-diffusers stable-video-diffusion-img2vid/vae xnemo_ckpt umt5-base; do chk "$XN_PRETRAINED/$d" "$d"; done
echo "== 起点 ckpt(按 exps/ 需要)"
for f in output/s1_uni/stage1_step_750.pt output/s2_L64ft/stage2_step_2000.pt; do chk "$XN_OUTPUT/${f#output/}" "$f"; done
echo "== Python 依赖($XN_PY)"
$XN_PY - <<'PY'
import importlib
for m in ["torch","diffusers","transformers","omegaconf","einops","lpips","pytorch_fid","skimage","cv2","mediapipe","scipy"]:
    try: v=getattr(importlib.import_module(m),"__version__","?"); print(f"  ✓ {m} {v}")
    except Exception as e: print(f"  ✗ {m}: {type(e).__name__} {e}")
import torch; print(f"  CUDA 可用={torch.cuda.is_available()} GPU 数={torch.cuda.device_count()}")
PY
[ -n "$XN_FACE_PY" ] && { $XN_FACE_PY -c "import insightface;print('  ✓ insightface(盲测)')" 2>/dev/null || echo "  ✗ insightface(盲测视频不可用,不影响训练)"; }
echo "== GPU"; nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /'
echo "== 磁盘"; df -h "$XN_OUTPUT" 2>/dev/null | tail -1 | sed 's/^/  /'
echo "== 结论:$ok 项通过,$bad 项缺失"
