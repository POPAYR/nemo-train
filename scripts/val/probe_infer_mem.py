"""部署态推理显存实测：reference_unet + denoising_unet(1685M) + TAESD，跑 4 步 8 帧 block。
回答"能否塞进 4090 24GB"。"""
import os, sys, time
import numpy as np, torch
XNEMO_ROOT = "/media/ps/ssd5/ayr/x-nemo-inference"
if XNEMO_ROOT not in sys.path: sys.path.append(XNEMO_ROOT)
from omegaconf import OmegaConf
from diffusers import AutoencoderTiny
from src.models.mutual_self_attention import ReferenceAttentionControl
from scripts.val.bench_decoder_phase0 import build_decoder


def main():
    dev = torch.device("cuda:0"); dt = torch.float16
    torch.backends.cudnn.benchmark = True
    cfg = OmegaConf.load(os.path.join(XNEMO_ROOT, "configs/test_ar_model.yaml"))
    ru, du, svd = build_decoder(cfg, dev, dt); du.eval()
    del svd; torch.cuda.empty_cache()   # 部署不用 SVD VAE，只用 TAESD
    vae = AutoencoderTiny.from_pretrained("/media/ps/ssd5/ayr/pretrained/taesd", torch_dtype=dt).to(dev).eval()
    torch.cuda.reset_peak_memory_stats()
    w_mem = torch.cuda.memory_allocated()/1e9
    print(f"[weights] reference+denoising(1685M)+TAESD 常驻 = {w_mem:.2f} GB")

    H = W = 64; F = 8
    w = ReferenceAttentionControl(ru, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
    r = ReferenceAttentionControl(du, do_classifier_free_guidance=False, mode="read", batch_size=1, fusion_blocks="full")
    ce = torch.randn(1,1,768,device=dev,dtype=dt); rl = torch.randn(1,4,H,W,device=dev,dtype=dt); t = torch.tensor(500,device=dev)
    lat = torch.randn(1,4,F,H,W,device=dev,dtype=dt); mot = torch.randn(1,F,32,16,device=dev,dtype=dt)

    @torch.no_grad()
    def block4step():
        w.clear(); ru(rl, torch.zeros_like(t), encoder_hidden_states=ce, return_dict=False); r.update(w)
        x = lat
        for _ in range(4):  # 4 步去噪
            x = du(x, t, encoder_hidden_states=[ce, mot], pose_cond_fea=None, return_dict=False)[0]
        img = vae.decode(x[0].permute(1,0,2,3)).sample  # [F,3,512,512]
        return img
    for _ in range(3): block4step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(10):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); block4step(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    peak = torch.cuda.max_memory_allocated()/1e9
    med = float(np.median(ts))
    print(f"[推理] 4步+8帧+TAESD 一个 block: {med:.1f} ms → {1000*F/med:.1f} fps")
    print(f"[峰值显存] 部署态推理 = {peak:.2f} GB   （4090=24GB, {'✅塞得下' if peak<24 else '❌超'})")


if __name__ == "__main__":
    main()
