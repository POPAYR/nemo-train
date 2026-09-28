import os
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torchvision.transforms as transforms
from transformers import UMT5EncoderModel, Wav2Vec2Model
from diffusers.video_processor import VideoProcessor
from dataset import VideoAudioTextDataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 显卡序号
    torch.cuda.set_device(3) 
    init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    destroy_process_group()

def main_worker(rank, world_size, config):
    setup(rank, world_size)
    device = torch.device(f"cuda:3") # 依然使用你指定的 cuda:3
    
    # 1. 加载模型
    text_encoder = UMT5EncoderModel.from_pretrained(config['text_encoder_path'])
    audio_encoder = Wav2Vec2Model.from_pretrained(config['audio_encoder_path'])
    video_processor = VideoProcessor(do_resize=True, vae_scale_factor=8)
    
    text_encoder.to(device).eval()
    audio_encoder.to(device).eval()
    
    # 2. 准备数据集 (使用 DistributedSampler)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = VideoAudioTextDataset(
        video_dir=config['video_dir'],
        audio_dir=config['audio_dir'],
        data_name_path=config['train_data_name_path'],
        caption_dir=config['caption_dir'],
        tokenizer_path=config['tokenizer_path'],
        transform=transform,
        frames=14,
        video_processor=video_processor,
        fps=25,
    )
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        sampler=sampler, 
        num_workers=4, # 每个进程分配 4 个 worker 加载数据
        pin_memory=True
    )
    
    # 3. 运行检查
    invalid_data_names = []
    
    # 仅在 rank 0 显示 tqdm 进度条
    pbar = tqdm(dataloader, disable=(rank != 0))
    
    for batch in pbar:
        # 将 Tensor 转移到 GPU
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        data_name = batch["data_name"][0]
        
        try:
            with torch.no_grad():
                # 模拟原始推理逻辑
                text_token = batch["text_token"]
                audio_clip = batch["audio_clip"]
                audio_window = batch["audio_window"]
                
                text_encoder(text_token)
                audio_encoder(audio_clip)
                
                B, T, S = audio_window.shape
                audio_encoder(audio_window.reshape(B*T, S))
        except Exception as e:
            invalid_data_names.append(data_name)

    # 4. 汇总结果
    # 每个进程先存个临时文件
    temp_file = f"temp_invalid_{rank}.txt"
    with open(temp_file, 'w') as f:
        for name in invalid_data_names:
            f.write(name + '\n')
            
    torch.distributed.barrier() # 等待所有进程完成

    # 由 rank 0 合并所有临时文件
    if rank == 0:
        with open(config['output_file'], 'w') as final_f:
            for i in range(world_size):
                t_file = f"temp_invalid_{i}.txt"
                if os.path.exists(t_file):
                    with open(t_file, 'r') as tf:
                        final_f.write(tf.read())
                    os.remove(t_file) # 删除临时文件
        print(f"Done! Invalid names saved to {config['output_file']}")

    cleanup()

if __name__ == "__main__":
    # 配置字典
    config = {
        'text_encoder_path': "/media/ps/ssd5/ayr/pretrained/umt5-base",
        'audio_encoder_path': "/media/ps/ssd5/ayr/pretrained/wav2vec2-base-960h",
        'video_dir': "/media/ps/ssd4/ayr/hallo3_frames_512/face_frames",
        'audio_dir': "/media/ps/ssd4/ayr/hallo3_frames_512/audio_wav",
        'train_data_name_path': "/media/ps/ssd4/ayr/hallo3_frames_512/valid_data.txt",
        'caption_dir': "/media/ps/ssd4/ayr/hallo3_frames_512/emo_pose_caption",
        'tokenizer_path': "/media/ps/ssd5/ayr/pretrained/umt5-base",
        'output_file': '/media/ps/ssd4/ayr/hallo3_frames_512/invalid.txt'
    }

    # 设置你想要的并行进程数（例如在一张卡上跑 4 个进程）
    # 进程数不宜过多，否则显存会炸，建议 2-4 个
    world_size = 4 
    mp.spawn(main_worker, args=(world_size, config), nprocs=world_size, join=True)