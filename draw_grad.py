import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_url
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        # これを有効にしないと、計算した勾配が毎回異なり、再現性が担保できない。
        torch.backends.cudnn.deterministic = True
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def predict(model,images):
    model.eval()

    pred = []
    for img in images:
        pred.append(model(img))

    return np.array(pred)

transform = transforms.Compose(
        [
            transforms.Resize(256),  # (256, 256) で切り抜く。
            transforms.CenterCrop(224),  # 画像の中心に合わせて、(224, 224) で切り抜く
            transforms.ToTensor(),  # テンソルにする。
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),  # 標準化する。
        ]
    )

if __name__ == '__main__':
    # デバイスを選択する。
    device = get_device(use_gpu=True)

    img_list = list(Path('../ILSVRC2012_img_val_color').glob('*.JPEG'))
    img_list.sort()

    #realを使って精度を評価する
    with open('../real.json') as f:
        labels = json.load(f)
        
    labels = np.array(labels)
    labels = labels[:100]
    img_list = img_list[:100]

    images = []
    for path in img_list:
        img = Image.open(path)
        if img.mode == 'L':
            img = np.array(img)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            img = Image.fromarray(img)
        img = transform(img)
        images.append(img.unsqueeze(0).to(device))

    vgg16 = torchvision.models.vgg16(pretrained=True).to(device)
    vgg16.eval()
    vgg16 

    resnet50 = torchvision.models.resnet50(pretrained=True).to(device)