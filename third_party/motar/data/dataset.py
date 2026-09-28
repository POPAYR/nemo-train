import os
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
import sys as _sys
if _REPO not in _sys.path: _sys.path.insert(0, _REPO)
from src.utils.paths import P as XP, third_party  # 跨机器路径,见 docs/decisions/2026-09-28_two_machine_workflow.md
import json
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer, CLIPImageProcessor


def read_list(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def check_exist(p):
    return os.path.exists(p)


class ShortClipError(RuntimeError):
    """样本可用帧数 < context_length。属于**预期内的跳过**,不是数据损坏。
    与真正的加载失败区分开,否则 __getitem__ 的重试会把损坏样本一起静默吞掉。"""
    pass


class MotarDataset(Dataset):
    """
    统一的 dataset，AR 和 SVD 两个阶段共用。

    共有字段（两阶段都返回）:
        motion_tensor: [ctx, motion_dim]   全局 stats 归一化
        audio_clip:    [ctx * spf]         原始波形（与 motion 同窗口对齐）
        text_token:    [text_max_len]
        text_mask:     [text_max_len]
        mask:          [ctx]               motion 有效帧 mask（pad_short=False 时恒为全1）

    SVD 额外字段（load_video=True 时返回）:
        video_tensor:  [ctx, C, H, W]      VAE latent
        ref_latent:    [C, H, W]           参考帧 VAE latent（窗口首帧）
        ref_img:       [3, 224, 224]       参考帧 CLIP 预处理图

    模式开关:
        pad_short:  True(AR)  短样本末帧 repeat 补齐 + mask；
                    False(SVD) 短样本直接报错（视频帧不能 pad）。
        random_crop:True 随机窗口（训练）；False 固定 start=0（验证，跨 step 可比）。
        load_video: True 加载 video latent / ref（SVD）；False 不加载（AR）。
    """

    def __init__(
        self,
        pose_dir,
        audio_dir,
        caption_dir,
        data_name_path,
        tokenizer_path,
        data_stats_path,
        context_length=128,
        fps=25,
        sr=16000,
        text_max_len=128,
        random_crop=True,
        pad_short=True,
        load_video=False,
        latent_dir=None,
        video_dir=None,
        pose_dir_alt=None,
        bbox_drop=0.0,
        video_processor=None,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.pose_dir = Path(pose_dir)
        # 可选的第二份 motion latent(常量 bbox 版),与 bbox_drop 配合使用;
        # 不传则行为与原来完全一致,不影响其他调用方。
        self.pose_dir_alt = Path(pose_dir_alt) if pose_dir_alt else None
        self.bbox_drop = bbox_drop
        self.audio_dir = Path(audio_dir)
        self.caption_dir = Path(caption_dir)
        self.data_list = read_list(data_name_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self.context_length = context_length
        self.fps = fps
        self.audio_sample_rate = sr
        self.samples_per_frame = sr // fps      # 16000 // 25 = 640
        self.text_max_len = text_max_len
        self.random_crop = random_crop
        self.pad_short = pad_short

        # SVD-only
        self.load_video = load_video
        if load_video:
            assert latent_dir is not None and video_dir is not None, \
                "load_video=True 需要 latent_dir 和 video_dir"
            self.latent_dir = Path(latent_dir)
            self.video_dir = Path(video_dir)
            self.video_processor = video_processor
            self.clip_image_processor = CLIPImageProcessor()

        # 全局 stats（AR / SVD / 未来联合训练共用同一份）
        data_stats = torch.load(data_stats_path, map_location="cpu")
        self.mean = data_stats["mean"].float().reshape(-1)   # [motion_dim]
        self.std = data_stats["std"].float().reshape(-1)


    def __len__(self):
        return len(self.data_list)

    # ---------------- CLIP 图像预处理（SVD ref image）----------------
    def _process_clip_image(self, img_tensor):
        from utils.processor import _resize_with_antialiasing
        img = img_tensor.unsqueeze(0)
        img = img * 2.0 - 1.0
        img = _resize_with_antialiasing(img, (224, 224))
        img = (img + 1.0) / 2.0
        return self.clip_image_processor(
            images=img.squeeze(0), return_tensors="pt",
            do_rescale=False, do_resize=False,
        ).pixel_values.squeeze(0)

    def getitem(self, index):
        data_name = self.data_list[index]
        # ★ bbox_drop:在"真实逐帧 bbox"与"常量 [0,0,1]"两份 motion latent 之间按概率选。
        #   背景运动本来没有任何条件信号(裁剪跟随人脸 ⇒ 背景平移,但模型收不到),
        #   导致背景乱动/边缘闪烁(见 FLOW_DISTILL_PROGRESS §5d)。
        #   训练喂真实值让模型学会"窗口移动⇒背景平移";推理喂常量表示"窗口不动"⇒静止背景。
        #   按 bbox_drop 概率混入常量版,是为了让"常量"这个推理输入也在训练分布内。
        pose_dir = self.pose_dir
        if self.pose_dir_alt is not None and random.random() < self.bbox_drop:
            pose_dir = self.pose_dir_alt
        pose_path = pose_dir / f"{data_name}.pt"
        audio_path = self.audio_dir / f"{data_name}.pt"     # 预处理后的波形 .pt
        caption_path = self.caption_dir / f"{data_name}.json"

        need = [pose_path, audio_path, caption_path]
        if self.load_video:
            latent_path = self.latent_dir / f"{data_name}.pt"
            video_folder = self.video_dir / f"{data_name}"
            need += [latent_path, video_folder]
        if not all(check_exist(p) for p in need):
            raise RuntimeError(f"Missing data for {data_name}")

        # -------- pose latent: [T, 32, 16] --------
        pose_video_tensor = torch.load(pose_path, map_location="cpu").squeeze(0)
        T_pose = pose_video_tensor.shape[0]

        # -------- video latent（SVD）--------
        video_latents = None
        if self.load_video:
            video_latents = torch.load(latent_path, map_location="cpu", weights_only=True)
            T_pose = min(T_pose, video_latents.shape[0])     # video 也参与公共长度

        # -------- audio waveform（预处理 .pt）--------
        audio = torch.load(audio_path, map_location="cpu").float()   # [N] 1D
        T_audio = audio.shape[0] // self.samples_per_frame
        audio = audio[: T_audio * self.samples_per_frame]

        # -------- 公共可用帧数 --------
        L = min(T_pose, T_audio)
        if L <= 0:
            raise RuntimeError(f"Empty usable length {data_name} (pose={T_pose}, audio={T_audio})")

        ctx = self.context_length
        if L >= ctx:
            start = np.random.randint(0, L - ctx + 1) if self.random_crop else 0
            idx = list(range(start, start + ctx))
            pose_clip = pose_video_tensor[start:start + ctx]
            mask = torch.ones(ctx)
            valid = ctx
        else:
            if not self.pad_short:
                # ★ 不足 ctx 帧 → 直接跳过,不做 pad。
                #   末帧 repeat 会让监督目标的后半段变成**完全静止的重复帧**,
                #   时序模块会在这些段上建模,学到病态的"后半段不动"模式。
                #   (loss 有 mask 不计入,但前向仍然看到,注意力照样被影响)
                raise ShortClipError(f"Not enough frames {data_name} (L={L} < ctx={ctx})")
            # AR：末帧 repeat 补齐 + mask
            start = 0
            idx = list(range(L)) + [L - 1] * (ctx - L)
            pad_len = ctx - L
            pose_clip = torch.cat(
                [pose_video_tensor[:L], pose_video_tensor[L - 1:L].repeat(pad_len, 1, 1)], dim=0)
            mask = torch.cat([torch.ones(L), torch.zeros(pad_len)])
            valid = L

        # flatten + 归一化
        pose_clip = pose_clip.flatten(1)                                # [ctx, motion_dim]
        pose_clip = (pose_clip - self.mean) / (self.std + 1e-6)

        # -------- 对齐音频段 [ctx * spf] --------
        spf = self.samples_per_frame
        audio_clip = audio[start * spf:(start + valid) * spf]
        target_len = ctx * spf
        if audio_clip.shape[0] < target_len:
            audio_clip = F.pad(audio_clip, (0, target_len - audio_clip.shape[0]))
        else:
            audio_clip = audio_clip[:target_len]
        audio_clip = audio_clip.contiguous()

        # -------- caption --------
        caption_dict = json.load(open(caption_path, "r", encoding="utf-8"))
        values = list(caption_dict.values())
        caption = values[0] if values else ""
        tok = self.tokenizer(
            caption, padding="max_length", truncation=True,
            max_length=self.text_max_len, return_tensors="pt")
        text_token = tok.input_ids.squeeze(0)
        text_mask = tok.attention_mask.squeeze(0)

        out = {
            "data_name": data_name,
            "motion_tensor": pose_clip,      # [ctx, motion_dim] 归一化
            "audio_clip": audio_clip,        # [ctx * spf]
            "text_token": text_token,
            "text_mask": text_mask,
            "mask": mask,
        }

        # -------- video 额外字段 --------
        if self.load_video:
            video_tensor = video_latents[idx]            # [ctx, C, H, W]（idx 已含 pad 重复）
            ref_idx = idx[0]
            out["video_tensor"] = video_tensor
            out["ref_latent"] = video_latents[ref_idx]
            # ref image（窗口首帧）
            frame_paths = sorted(self.video_dir.joinpath(data_name).glob("*.jpg"))
            ref_pil = Image.open(frame_paths[ref_idx]).convert("RGB")
            ref_np = self.video_processor.pil_to_numpy(ref_pil)
            ref_pt = self.video_processor.numpy_to_pt(ref_np).squeeze(0)
            out["ref_img"] = self._process_clip_image(ref_pt)
            # motion_encoder 用的参考帧预处理（VaeImageProcessor，和 X-Nemo 推理 neg_motion 一致）
            if not hasattr(self, "_cond_proc"):
                from diffusers.image_processor import VaeImageProcessor
                self._cond_proc = VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True, do_normalize=True)
            out["ref_mot_cond"] = self._cond_proc.preprocess(ref_pil, height=224, width=224).squeeze(0)

        return out

    _n_short = 0
    _n_bad = 0

    def __getitem__(self, index):
        last_err = None
        for _ in range(20):
            try:
                return self.getitem(index)
            except ShortClipError as e:
                # 预期内:换一个样本,不算错误
                last_err = e
                MotarDataset._n_short += 1
                index = np.random.randint(len(self.data_list))
            except Exception as e:
                # ★ 真正的加载失败必须可见 —— 早先版本把两类一起静默重试,
                #   数据损坏会被伪装成"正常训练"。
                last_err = e
                MotarDataset._n_bad += 1
                if MotarDataset._n_bad <= 20 or MotarDataset._n_bad % 500 == 0:
                    print(f"[dataset:WARN] 加载失败 #{MotarDataset._n_bad}: {type(e).__name__}: {e}", flush=True)
                index = np.random.randint(len(self.data_list))
        raise RuntimeError(f"连续 20 次取样失败(short={MotarDataset._n_short} bad={MotarDataset._n_bad}), "
                           f"最后一个错误: {last_err}")


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from diffusers.video_processor import VideoProcessor
    video_processor = VideoProcessor(do_resize=True, vae_scale_factor=8)
    ds = MotarDataset(
        pose_dir=XP("XN_HALLO3", "pose_embed"),
        audio_dir=XP("XN_HALLO3", "audio_pt"),
        latent_dir=XP("XN_HALLO3", "frame_latent"),
        caption_dir=XP("XN_HALLO3", "emo_pose_caption"),
        data_name_path=XP("XN_HALLO3", "valid_data.txt"),
        tokenizer_path=XP("XN_PRETRAINED", "umt5-base"),
        data_stats_path=os.path.join(third_party("motar"), "global_motion_latent_stats.pt"),
        context_length=14,
        random_crop=False,
        pad_short=False,
        load_video=True,
        video_processor=video_processor,
    )
    sample = ds[0]
    torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k,v in sample.items()},
            "/media/ps/ssd5/ayr/motar/overfit_sample.pt")
    print({k: (v.shape if torch.is_tensor(v) else type(v)) for k,v in sample.items()})
    dataloader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=64)
    for i, batch in enumerate(dataloader):
    # for batch in tqdm(dataloader):
        print(f"batch{i}:")
        for key, value in batch.items():
            try:
                print(f"{key}: {value.shape}")
            except:
                print(f"{key}: {value}")
        if i >= 2:
            break