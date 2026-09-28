import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import (
    UMT5EncoderModel, 
    Wav2Vec2Model, 
    Wav2Vec2Processor,
    CLIPVisionModelWithProjection, 
    CLIPImageProcessor
    )
from diffusers.video_processor import VideoProcessor
from dataset import VideoAudioTextDataset

if __name__ == "__main__":
    device = "cuda:3"
    dtype = torch.bfloat16
    
    text_encoder_path = "/media/ps/ssd5/ayr/pretrained/umt5-base"
    audio_encoder_path = "/media/ps/ssd5/ayr/pretrained/wav2vec2-base-960h"
    video_dir = "/media/ps/ssd4/ayr/hallo3_frames_512/face_frames"
    audio_dir = "/media/ps/ssd4/ayr/hallo3_frames_512/audio_wav"
    train_data_name_path = "/media/ps/ssd4/ayr/hallo3_frames_512/valid_data.txt"
    caption_dir = "/media/ps/ssd4/ayr/hallo3_frames_512/emo_pose_caption"
    tokenizer_path = "/media/ps/ssd5/ayr/pretrained/umt5-base"
    
    output_file = '/media/ps/ssd4/ayr/hallo3_frames_512/invalid.txt'
    
    text_encoder = UMT5EncoderModel.from_pretrained(text_encoder_path)
    audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_path)
    video_processor = VideoProcessor(do_resize=True, vae_scale_factor=8)
    
    text_encoder.requires_grad_(False).to(device, dtype=torch.float32)
    audio_encoder.requires_grad_(False).to(device, dtype=torch.float32)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_dataset = VideoAudioTextDataset(
        video_dir=video_dir,
        audio_dir=audio_dir,
        data_name_path=train_data_name_path,
        caption_dir=caption_dir,
        tokenizer_path=tokenizer_path,
        transform=transform,
        frames=14,
        video_processor=video_processor,
        fps=25,
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=1,
        num_workers=8,
        pin_memory=True,
    )
    
    invalid_data_names = []
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        ref_img = batch["ref_img"]  # [B, 3, 512, 512]
        video_tensor = batch["video_tensor"]  # [B, T, 3, 512, 512]
        audio_window = batch["audio_window"]  # [B, T, 5*640]
        audio_clip = batch["audio_clip"]  # [B, T*640]
        text_token = batch["text_token"]  # [B, 128]
        data_name = batch["data_name"][0]
        
        with torch.no_grad():
            try:
                # Text and audio Embedding (fp32)
                text_emb = text_encoder(text_token).last_hidden_state                      
                audio_emb = audio_encoder(audio_clip).last_hidden_state
                
                # local audio feature
                B, T, S = audio_window.shape
                audio_feat = audio_encoder(audio_window.reshape(B*T, S)).last_hidden_state
                local_audio_emb = audio_feat.reshape(B, T, audio_feat.shape[1], audio_feat.shape[2])
            except:
                invalid_data_names.append(data_name)
                
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for name in invalid_data_names:
            outfile.write(name + '\n')