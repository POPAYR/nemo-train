"""跨机器路径:所有数据/权重/测试集根目录都从环境变量取,缺省值 = 本机(开发机)路径。

另一台机器(实验机)在 configs/paths/<名字>.env 里覆盖这些变量,由 remote/*.sh 自动 source。
yaml 里用 OmegaConf 的 ${oc.env:XN_HALLO3,<本机默认>} 写法,与这里同名同默认。
决策记录:docs/decisions/2026-09-28_two_machine_workflow.md
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULTS = {
    "XN_PRETRAINED": "/media/ps/ssd5/ayr/pretrained",                 # sd-image-variations / SVD vae / xnemo_ckpt / umt5 ...
    "XN_HALLO3":     "/media/ps/ssd4/ayr/hallo3_frames_512",          # frame_latent / pose_embed_real / face_frames / audio_*
    "XN_HALLO3_RAW": "/media/ps/ssd5/ayr/hallo3-data/videos",      # 原始 mp4(实验机从这里重新处理出 XN_HALLO3)
    "XN_MEAD":       "/media/ps/ssd4/ayr/mead_fixed",
    "XN_TESTSET":    "/media/ps/ssd5/ayr/eval_metrics/testset",       # manifest.json / hallo3_subset30.json
    "XN_HALLO3_TEST": "/media/ps/ssd4/ayr/hallo3_test",
    "XN_FVD_I3D":    "/media/ps/ssd5/ayr/eval_metrics/fvd/cmvq2/fvd/styleganv/i3d_torchscript.pt",
    "XN_OUTPUT":     os.path.join(REPO, "output"),
}


def P(key: str, *parts: str) -> str:
    """XP("XN_HALLO3", "frame_latent") → <根>/frame_latent(调用方统一 import 为 XP,避免与局部变量 P 撞名)。未知 key 直接报错,避免静默拼出错路径。"""
    if key == "REPO":
        base = REPO
    elif key in DEFAULTS:
        base = os.environ.get(key, DEFAULTS[key])
    else:
        raise KeyError(f"未知路径键 {key};可用:{['REPO'] + list(DEFAULTS)}")
    return os.path.join(base, *parts)


# 开发机路径 → 当前机器路径的前缀映射(测试集 manifest.json 等数据文件里写死了开发机绝对路径)
_DEV_PREFIX = {
    "/media/ps/ssd5/ayr/eval_metrics/testset": "XN_TESTSET",
    "/media/ps/ssd4/ayr/hallo3_frames_512": "XN_HALLO3",
    "/media/ps/ssd5/ayr/hallo3-data/videos": "XN_HALLO3_RAW",
    "/media/ps/ssd4/ayr/mead_fixed": "XN_MEAD",
    "/media/ps/ssd5/ayr/pretrained": "XN_PRETRAINED",
}


def remap(path: str) -> str:
    """把数据文件里记录的开发机绝对路径换成当前机器的对应路径;开发机上原样返回。"""
    for pre, key in _DEV_PREFIX.items():
        if path == pre or path.startswith(pre + "/"):
            return P(key) + path[len(pre):]
    return path


def third_party(name: str) -> str:
    """仓库内置的第三方代码目录(motar 数据加载、FVD),用于 sys.path.insert。"""
    return os.path.join(REPO, "third_party", name)


if __name__ == "__main__":        # python -m src.utils.paths → 打印当前生效的全部路径及是否存在
    for k in ["REPO"] + list(DEFAULTS):
        v = P(k)
        print(f"{'✓' if os.path.exists(v) else '✗'} {k:15s} {v}")
