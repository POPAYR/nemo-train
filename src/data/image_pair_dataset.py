"""
Stage 1 专用:跨帧重建(cross-reconstruction)图像对数据集
========================================================
与 MotarDataset 的关键区别 —— **参考帧与目标帧都从整段视频里随机抽**,不受滑动窗口限制:
  MotarDataset:  参考=窗口首帧,目标=其后连续 L 帧  → 姿态差被窗口长度封死,且同 batch 内高度相关
  本 dataset:    参考=随机帧,目标=另外随机 n_target 帧 → 姿态/表情差可跨整段视频,泛化性更好

只为「图像 backbone 的 flow 适配」服务,因此:
  - 不加载音频/caption(stage1 用不到)
  - motion 用**预计算的 pose latent**(motion encoder 天然冻结),直接给原始值,不做 stats 归一化
    (UNet 吃的就是原始 motion token;MotarDataset 那边是归一化后再由训练脚本还原,这里省掉往返)

返回:
  ref_latent   [C,H,W]          参考帧 VAE latent
  ref_img      [3,224,224]      参考帧 CLIP 预处理图
  tgt_latent   [N,C,H,W]        目标帧 VAE latent
  tgt_motion   [N,32,16]        目标帧 motion token(原始值)
"""
import os, random
from pathlib import Path
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


def _read_list(p):
    with open(p, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


class ImagePairDataset(Dataset):
    def __init__(self, pose_dir, latent_dir, video_dir, data_name_path, pose_dir_alt=None, bbox_drop=0.0,
                 n_target=4, min_gap=0, video_processor=None):
        self.pose_dir = Path(pose_dir); self.latent_dir = Path(latent_dir); self.video_dir = Path(video_dir)
        # ★ bbox_drop:在真实/常量两份 motion latent 间按概率选(见 FLOW_DISTILL_PROGRESS §5d)
        self.pose_dir_alt = Path(pose_dir_alt) if pose_dir_alt else None
        self.bbox_drop = bbox_drop
        self.data_list = _read_list(data_name_path)
        self.n_target = n_target
        self.min_gap = min_gap          # 目标帧与参考帧的最小间隔(0=不限制)
        self.clip_image_processor = CLIPImageProcessor()
        self.video_processor = video_processor

    def __len__(self):
        return len(self.data_list)

    def _process_clip_image(self, img_tensor):
        from utils.processor import _resize_with_antialiasing
        img = img_tensor.unsqueeze(0)
        img = img * 2.0 - 1.0
        img = _resize_with_antialiasing(img, (224, 224))
        img = (img + 1.0) / 2.0
        return self.clip_image_processor(images=img.squeeze(0), return_tensors="pt",
                                         do_rescale=False, do_resize=False).pixel_values.squeeze(0)

    def __getitem__(self, index):
        for _ in range(8):                      # 坏样本重采
            try:
                return self._get(index)
            except Exception:
                index = random.randrange(len(self.data_list))
        raise RuntimeError("连续 8 次取样失败")

    def _get(self, index):
        name = self.data_list[index]
        lat = torch.load(self.latent_dir / f"{name}.pt", map_location="cpu", weights_only=True)  # [T,C,H,W]
        _pd = self.pose_dir_alt if (self.pose_dir_alt is not None and random.random() < self.bbox_drop) else self.pose_dir
        pose = torch.load(_pd / f"{name}.pt", map_location="cpu").squeeze(0)                     # [T,32,16]
        T = min(lat.shape[0], pose.shape[0])
        if T < 2:
            raise RuntimeError(f"too short: {name} T={T}")

        # ★ 参考帧、目标帧均从**整段视频**随机抽
        ref_idx = random.randrange(T)
        if self.min_gap > 0:
            cand = [i for i in range(T) if abs(i - ref_idx) >= self.min_gap] or list(range(T))
        else:
            cand = list(range(T))
        tgt_idx = [random.choice(cand) for _ in range(self.n_target)]

        frame_paths = sorted(self.video_dir.joinpath(name).glob("*.jpg"))
        if len(frame_paths) <= ref_idx:
            raise RuntimeError(f"frames missing: {name}")
        ref_pil = Image.open(frame_paths[ref_idx]).convert("RGB")
        ref_np = self.video_processor.pil_to_numpy(ref_pil)
        ref_pt = self.video_processor.numpy_to_pt(ref_np).squeeze(0)

        return {
            "ref_latent": lat[ref_idx].float(),
            "ref_img": self._process_clip_image(ref_pt),
            "tgt_latent": lat[tgt_idx].float(),      # [N,C,H,W]
            "tgt_motion": pose[tgt_idx].float(),     # [N,32,16] 原始 motion token
            "gap": torch.tensor([abs(i - ref_idx) for i in tgt_idx], dtype=torch.float32),
        }
