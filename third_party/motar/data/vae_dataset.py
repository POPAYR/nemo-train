import os
import random
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

camera_angle_list = ['down', 'front', 'left_30', 'left_60', 'right_30', 'right_60', 'top']
emotion_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
level_list = ['level_1', 'level_2', 'level_3']

def read_list(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

class VAEDatasetCross(Dataset):
    def __init__(self, mead_list, mead_video_dir, hallo_list, hallo_video_dir, transform=None):
        self.mead_list = read_list(mead_list)
        self.hallo_list = read_list(hallo_list)
        self.mead_video_dir = Path(mead_video_dir)
        self.hallo_video_dir = Path(hallo_video_dir)
        self.transform = transform

    def __len__(self):
        return max(len(self.mead_list), len(self.hallo_list))

    def __getitem__(self, index):
        '''
        Returns a dictionary containing:
        mead_data: npz data from MEAD dataset
        cross_mead_data: npz data from MEAD dataset with different conditions
        mead_img: corresponding image from MEAD dataset
        cross_mead_img: corresponding image from MEAD dataset with different conditions
        hallo_data: npz data from Hallo dataset
        hallo_img: corresponding image from Hallo dataset
        '''
        mead_index = index % len(self.mead_list)
        hallo_index = index % len(self.hallo_list)

        mead_path = self.mead_list[mead_index]
        hallo_path = self.hallo_list[hallo_index]

        mead_npz_list = [f for f in os.listdir(mead_path) if os.path.isfile(os.path.join(mead_path, f))] # npz path list
        hallo_npz_list = [f for f in os.listdir(hallo_path) if os.path.isfile(os.path.join(hallo_path, f))]
        
        # ====== Process MEAD data ======
        mead_npz_path = os.path.join(mead_path, random.choice(mead_npz_list)) # /{train or val}/{person_id}/landmark/{camera_angle}/{emotion}/{level}/{video_id}/{frame_id}.npz
        mead_npz_path = Path(mead_npz_path)
        camera_angle, emotion, level, video_id= mead_npz_path.parts[-5:-1]
        person_id = mead_npz_path.parts[-7]
        mead_person_path = mead_npz_path.parents[5]  # parents[0] 是上一层，parents[4] 是上五层

        cross_npz_path = None
        while True:
            try:
                new_camera_angle = random.choice(camera_angle_list)
                new_emotion = random.choice(emotion_list)
                if new_emotion == 'neutral':
                    new_level = 'level_1'
                else:
                    new_level = random.choice(level_list)
                # randomly set a new frame with the same person but different cross condition
                mead_cross_path = mead_person_path / "landmark" / new_camera_angle / new_emotion / new_level
                # randomly select a video
                cross_video = random.choice([d for d in mead_cross_path.iterdir() if d.is_dir()])
                # randomly select a frame
                cross_npz_path = random.choice([f for f in cross_video.iterdir() if f.is_file() and f.suffix == ".npz"])
                break
            except Exception as e:
                continue

        mead_img_path = self.mead_video_dir / person_id / "video" / camera_angle / emotion / level / video_id / (mead_npz_path.stem + ".png")
        cross_mead_img_path = self.mead_video_dir / person_id / "video" / new_camera_angle / new_emotion / new_level / cross_video.name / (cross_npz_path.stem + ".png")

        mead_img = Image.open(mead_img_path).convert("RGB")
        cross_mead_img = Image.open(cross_mead_img_path).convert("RGB")

        if self.transform:
            mead_img = self.transform(mead_img)
            cross_mead_img = self.transform(cross_mead_img)
        
        mead_data = np.load(mead_npz_path)
        cross_mead_data = np.load(cross_npz_path)

        # ===== Process Hallo data ======
        hallo_npz_path = os.path.join(hallo_path, random.choice(hallo_npz_list))
        video_id, frame_id = Path(hallo_npz_path).parts[-2:]
        hallo_img_path = self.hallo_video_dir / video_id / (Path(hallo_npz_path).stem + ".png")
        hallo_img = Image.open(hallo_img_path).convert("RGB")

        if self.transform:
            hallo_img = self.transform(hallo_img)
        
        hallo_data = np.load(hallo_npz_path)

        return {
            'mead_data': mead_data, # npz dict keys: landmark_2d_106, landmark_3d_68, pose, id_embedding
            'cross_mead_data': cross_mead_data,
            'mead_img': mead_img, # torch.Size([b, 3, 256, 256])
            'cross_mead_img': cross_mead_img,
            'hallo_data': hallo_data,
            'hallo_img': hallo_img,
        }

class VAEDataset(Dataset):
    def __init__(self, hallo_list, hallo_video_dir, transform=None):
        self.hallo_list = read_list(hallo_list)
        self.hallo_video_dir = Path(hallo_video_dir)
        self.transform = transform

    def __len__(self):
        return len(self.hallo_list)

    def __getitem__(self, index):
        """
        Randomly sample two frames from the same video.
        Prefer temporal distance >= 30 frames;
        fallback to random two frames if not possible.
        """
        hallo_index = index % len(self.hallo_list)
        hallo_path = self.hallo_list[hallo_index]

        hallo_npz_list = [
            f for f in os.listdir(hallo_path)
            if os.path.isfile(os.path.join(hallo_path, f))
        ]

        # ===== parse frame ids =====
        frame_infos = []
        for f in hallo_npz_list:
            stem = Path(f).stem
            frame_id = int("".join(filter(str.isdigit, stem)))
            frame_infos.append((f, frame_id))

        assert len(frame_infos) >= 2, \
            f"Not enough frames in {hallo_path}"

        frame_infos.sort(key=lambda x: x[1])

        # ===== try to sample with distance >= 30 =====
        valid_pairs = []
        for i in range(len(frame_infos)):
            for j in range(i + 1, len(frame_infos)):
                if frame_infos[j][1] - frame_infos[i][1] >= 30:
                    valid_pairs.append((frame_infos[i], frame_infos[j]))

        if len(valid_pairs) > 0:
            (npz_name_1, frame_id_1), (npz_name_2, frame_id_2) = random.choice(valid_pairs)
        else:
            # ===== fallback: random two frames =====
            (npz_name_1, frame_id_1), (npz_name_2, frame_id_2) = \
                random.sample(frame_infos, 2)

        # ===== paths =====
        npz_path_1 = os.path.join(hallo_path, npz_name_1)
        npz_path_2 = os.path.join(hallo_path, npz_name_2)

        video_id = Path(hallo_path).name
        img_path_1 = self.hallo_video_dir / video_id / (Path(npz_name_1).stem + ".png")
        img_path_2 = self.hallo_video_dir / video_id / (Path(npz_name_2).stem + ".png")

        # ===== load =====
        hallo_img_1 = Image.open(img_path_1).convert("RGB")
        hallo_img_2 = Image.open(img_path_2).convert("RGB")

        if self.transform:
            hallo_img_1 = self.transform(hallo_img_1)
            hallo_img_2 = self.transform(hallo_img_2)

        hallo_data_1 = np.load(npz_path_1)
        hallo_data_2 = np.load(npz_path_2)

        return {
            "hallo_data_1": hallo_data_1,
            "hallo_data_2": hallo_data_2,
            "hallo_img_1": hallo_img_1,
            "hallo_img_2": hallo_img_2,
            "frame_id_1": frame_id_1,
            "frame_id_2": frame_id_2,
        }



if __name__ == "__main__":
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # dataset = VAEDatasetCross(
    #     mead_list='/media/ps/ssd5/ayr/MEAD-256-landmark/MEAD-train-data.txt',
    #     mead_video_dir='/media/ps/ssd5/ayr/MEAD-256',
    #     hallo_list='/media/ps/ssd5/ayr/hallo3-landmark/hallo3-train-data.txt',
    #     hallo_video_dir='/media/ps/ssd5/ayr/hallo3-data-256',
    #     transform=transform,
    # )

    # # 创建 DataLoader
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    # print("Dataset length:", len(dataset))

    # # 遍历几个 batch 验证
    # for i, batch in enumerate(dataloader):
    #     print(f"Batch {i}:")
    #     print("mead_img:", batch['mead_img'].shape)
    #     print("cross_mead_img:", batch['cross_mead_img'].shape)
    #     print("hallo_img:", batch['hallo_img'].shape)
    #     print("mead_data:", batch['mead_data'].keys()) 
    #     print("cross_mead_data:", batch['cross_mead_data'].keys())
    #     print("hallo_data:", batch['hallo_data'].keys())
    #     print(batch['hallo_data']['landmark_3d_68'].shape)
    #     if i >= 1:  # 只验证前两个 batch
    #         break
        
    dataset = VAEDataset(
        hallo_list='/media/ps/ssd5/ayr/hallo3-landmark/hallo3-train-data.txt',
        hallo_video_dir='/media/ps/ssd5/ayr/hallo3-data-256',
        transform=transform,
    )

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    print("Dataset length:", len(dataset))
    # 遍历几个 batch 验证
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print("hallo_img_1:", batch['hallo_img_1'].shape)
        print("hallo_img_2:", batch['hallo_img_2'].shape)
        print("hallo_data_1:", batch['hallo_data_1'].keys())
        print("hallo_data_2:", batch['hallo_data_2'].keys())
        print(batch['hallo_data_1']['landmark_3d_68'].shape)
        print(batch['h  allo_data_2']['landmark_3d_68'].shape)
        print("frame_id_1:", batch['frame_id_1'])
        print("frame_id_2:", batch['frame_id_2'])
        if i >= 1:  # 只验证前两个 batch
            break
        

