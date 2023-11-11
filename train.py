import torch
import json
import urllib
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
import pandas as pd
import cv2
import os


device = "GPU"


side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 20
sampling_rate = 10
frames_per_second = 30


transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size=(crop_size, crop_size))
        ]
    ),
)

# The duration of the input clip is also specific to the model.
clip_duration = (num_frames * sampling_rate)/frames_per_second
start_sec = 0
end_sec = start_sec + clip_duration


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, classes, transforms=None, max_frames= None):

        self.data_path = data_path
        self.classes = classes
        self.transforms = transforms
        self.allfiles = []
        for classes in os.listdir(data_path):
            if not os.path.isfile(classes):
                deep_dir = os.path.join(data_path,classes)
                for j in os.listdir(deep_dir):
                    # self.allfiles.append(deep_dir+'/'+j)
                    try: #костыль - исправление. Если ошибка вылетает, просто пропускаем этот файл
                        video = EncodedVideo.from_path(deep_dir+'/'+j)
                        self.allfiles.append(deep_dir+'/'+j)
                    except:
                        pass
        self.max_frames = max_frames

    def __len__(self):
        return len(self.allfiles)

    def read_video(self, path):

        video = EncodedVideo.from_path(path)

        # Load the desired clip
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

        # Apply a transform to normalize the video input
        video_data = self.transforms(video_data)
        # Move the inputs to the desired device
        inputs = video_data["video"]
        return inputs

    def __getitem__(self, idx) :
        v_path = self.allfiles[idx]
        label = v_path.split('/')[-2]
        return self.read_video(v_path), self.classes.index(label)
    


classes = pd.read_csv('classes.csv')

arra = []
for row in classes.iterrows():
    print(row[1][1])
    arra.append(row[1][1])


DataSet = CustomDataset('drive/MyDrive/hak/train_dataset_train/videos',classes=arra, transforms = transform)