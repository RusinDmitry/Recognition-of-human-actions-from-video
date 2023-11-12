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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory


if __name__ == "__main__":
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
                            video = EncodedVideo.from_path(deep_dir+'\\'+j)
                            self.allfiles.append(deep_dir+'\\'+j)
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
            label = v_path.split('\\')[-2]
            return self.read_video(v_path), self.classes.index(label)
        


    classes = pd.read_csv('classes.csv')

    arra = []
    for row in classes.iterrows():
        print(row[1][1])
        arra.append(row[1][1])


    DataSet = CustomDataset('videos',classes=arra, transforms = transform)


    dataloaders = torch.utils.data.DataLoader(DataSet, batch_size=4)

    dataset_sizes = len(DataSet)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def train_model(model, criterion, optimizer, scheduler = None, num_epochs=25):
        since = time.time()

        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

            torch.save(model.state_dict(), best_model_params_path)
            best_acc = 0.0

            for epoch in range(num_epochs):
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in dataloaders:
                        inputs = inputs.to(device)
                        print(labels)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        print("step2")
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels)
                    #if phase == 'train':
                    #    print("step")
                    #    scheduler.step()

                    epoch_loss = running_loss / dataset_sizes
                    epoch_acc = running_corrects.double() / dataset_sizes

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)

                print()

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:4f}')

            # load best model weights
            model.load_state_dict(torch.load(best_model_params_path))
        return model


    model_ft = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    # Set to GPU or CPU

    model_ft = model_ft.to(device)
    num_ftrs = model_ft.blocks[5].proj.in_features
    model_ft.blocks[5].proj = nn.Linear(num_ftrs, 24)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    model_ft = train_model(model_ft, criterion, optimizer_ft,
                        num_epochs=20)

    torch.save(model_ft, 'model_ft_pc20.pt')


    model_ft = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    # et to GPU or CPU

    model_ft = model_ft.to(device)
    num_ftrs = model_ft.blocks[5].proj.in_features
    model_ft.blocks[5].proj = nn.Linear(num_ftrs, 24)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    model_ft = train_model(model_ft, criterion, optimizer_ft,
                        num_epochs=10)

    torch.save(model_ft, 'model_ft_pc10.pt')