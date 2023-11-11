from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
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
import torch
import shutil

import os


# Загрузите вашу модель
model = torch.load("good_model.pt", map_location=torch.device("cpu"))
model.eval()

# Определите преобразования, которые необходимы для подготовки видео к входу модели
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Создайте экземпляр FastAPI
app = FastAPI()

# Создайте экземпляр Jinja2 Templates
templates = Jinja2Templates(directory="templates")

# Определите точку входа для отображения главной страницы
@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Определите точку входа для загрузки видеофайла и обработки его моделью
@app.post("/upload-video/")
async def upload_video(request: Request, file: UploadFile = File(...)):



    

    model = torch.load('good_model.pt')


    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 8
    sampling_rate = 8
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

    clip_duration = (num_frames * sampling_rate)/frames_per_second
    
    
    start_sec = 0
    end_sec = start_sec + clip_duration

    content = await file.read()

    with open("file.avi", "wb") as f:
        f.write(content)


    video = EncodedVideo.from_path(f'file.avi')
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
    video_data = transform(video_data)
    inputs = video_data["video"]


    
    json_filename = "kinetics_classnames.json"

    with open(json_filename, "r") as f:
        kinetics_classnames = json.load(f)

    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', "")
    
    preds = model(inputs[None, ...])

    # Get the predicted classes
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=5).indices[0]
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes]
    


    file_path = "file.avi"

    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Файл {file_path} успешно удален.")
    else:
        print(f"Файл {file_path} не существует.")

    # TODO: Преобразуйте видео и получите результат

    result = "Текстовый результат вашей модели"
    return templates.TemplateResponse("index.html", {"request": request, "result": pred_class_names})