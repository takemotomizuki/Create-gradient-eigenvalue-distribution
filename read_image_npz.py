from os import path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path

def read_image():

    # 設定パラメータ
    x = []        # 画像データ用
    y = []        # ラベルデータ用

    # ファイルの保存先を指定（ファイルの拡張子は.npz）
    outfile ="../dataset/imagenet_val_float_"

    # path 以下の画像を読み込む
    paths = list(Path('../ILSVRC2012_img_val').glob('*.JPEG'))
    paths.sort()

    # label を読み込む
    df = pd.read_csv("../convert_class_index.csv",
                   encoding="cp932",
                   )
    labels = df['PredictionString'].values
    labels = np.eye(1000)[labels]
  
     # 各ファイルを処理
    for path,label in zip(paths,labels):
        # 画像ファイルを読む
        img = Image.open(path)    # Pillow(PIL)で画像読込み
    
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize((224, 224))  # 画像サイズを224px × 224pxにする
        img = np.array(img,'f')  # ndarray化

        # 画像データ(img)とラベルデータ(label)をx, y のそれぞれのリストに保存
        x.append(img)
        y.append(label)

        ### ファイルへ保存 ###
        # npzで作成する場合
        if len(x)%1000 == 0:
            print(len(x))

    np.savez("../dataset/imagenet_val_float.npz", x=x, y=y)

if __name__ == '__main__':
    read_image()