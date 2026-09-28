"""双流视频渲染加速基准（只加载一次，原地切 compile）。隔离 video block 渲染瓶颈。
用法: SMOKE_DEV=cuda:6 python -m src.distill.bench_stream
"""
import os, time, statistics, torch
from .stream_pipeline import build_pipeline, enable_fast_math

DEV = torch.device(os.environ.get("SMOKE_DEV", "cuda:6"))
DT = torch.bfloat16
GEN_CKPT = "/media/ps/ssd5/ayr/x-nemo-inference/output/dmd2_win24_0701/ckpt/dmd2_step_26000.pt"


def run_blocks(vs, n, tag, skip):
    """逐块渲染+解码，打印每块耗时，返回稳态(skip 之后)中位数。"""
    ref_latent = torch.randn(1, 4, 64, 64, device=DEV, dtype=DT)
    ref_img = torch.randn(1, 3, 224, 224, device=DEV, dtype=DT)
    vs.set_reference(ref_latent, ref_img)                    # 重置 KV/offset
    dts = []
    for b in range(n):
        mot = torch.randn(8, 512, device=DEV, dtype=DT)
        torch.cuda.synchronize(DEV); t0 = time.perf_counter()
        x0 = vs.render_block(mot); pix = vs.decode_pixels(x0)
        torch.cuda.synchronize(DEV); dt = time.perf_counter() - t0
        dts.append(dt)
        print(f"  [{tag}] blk {b:2d}  {dt*1000:8.1f} ms", flush=True)
    steady = dts[skip:]
    med = statistics.median(steady)
    print(f"==> [{tag}] steady median = {med*1000:.1f} ms/block = {8/med:.1f} fps "
          f"(skip{skip}, min {min(steady)*1000:.1f}/max {max(steady)*1000:.1f})\n", flush=True)
    return med


if __name__ == "__main__":
    import sys; sys.path.append("/media/ps/ssd5/ayr/motar")
    from model.armodel import MotionTransformer
    print(f"device={DEV} dtype={DT}\n", flush=True)
    enable_fast_math()

    mm = MotionTransformer(motion_dim=512, audio_dim=768, depth=8, heads=8, dim_head=64,
                           max_len=128, diffloss_dim=512, diffloss_depth=2,
                           num_sampling_steps=50).to(DEV, DT).eval()
    pipe = build_pipeline(mm, DEV, DT, gen_ckpt=GEN_CKPT, block_size=8, fast_math=True)
    vs = pipe.vstream
    res = {}

    # 1) eager（fast_math + sdpa flash 已在 render 内）
    res["eager"] = run_blocks(vs, 12, "eager", skip=4)

    # 2) + compile generator（原地替换；逐块看是否因 KV 增长每块重编译）
    try:
        vs.M.generator = torch.compile(vs.M.generator, dynamic=True, fullgraph=False,
                                       mode="reduce-overhead")
        print("[compile] generator compiled (dynamic, reduce-overhead)\n", flush=True)
        res["compile_gen"] = run_blocks(vs, 20, "cgen", skip=10)   # 多跑，给编译预热留余量
    except Exception as e:
        print(f"[compile] generator FAILED: {repr(e)[:300]}\n", flush=True)

    # 3) + compile vae.decode
    try:
        vs.vae.decode = torch.compile(vs.vae.decode, fullgraph=False, mode="reduce-overhead")
        print("[compile] vae.decode compiled\n", flush=True)
        res["compile_gen+vae"] = run_blocks(vs, 16, "call", skip=8)
    except Exception as e:
        print(f"[compile] vae FAILED: {repr(e)[:300]}\n", flush=True)

    print("=================== 汇总 ===================", flush=True)
    b0 = res["eager"]
    for k, v in res.items():
        print(f"  {k:20s} {v*1000:8.1f} ms  {8/v:5.1f} fps  x{b0/v:.2f}", flush=True)
