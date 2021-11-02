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
    outfile ="../dataset/imagenet_val_float_1000.npz"

    # path 以下の画像を読み込む
    files = list(Path('../ILSVRC2012_img_val').glob('*.JPEG'))
    files.sort()
    files = files[:1000]
    # label を読み込む
    df = pd.read_csv("../convert_class_index.csv",
                   encoding="cp932",
                   )
    labels = df['PredictionString'].values
    labels = labels[:1000]
  
     # 各ファイルを処理
    for f,label in zip(files,labels):
        # 画像ファイルを読む
        img_pil = Image.open(f)    # Pillow(PIL)で画像読込み。色順番はRGB
        img = np.array(img_pil,'f')  # ndarray化
        if img_pil.mode == 'L':
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (224, 224), cv2.INTER_LANCZOS4)  # 画像サイズを224px × 224pxにする
            
        # 画像データ(img)とラベルデータ(label)をx, y のそれぞれのリストに保存
        x.append(img)
        y.append(label)

    ### ファイルへ保存 ###
    # npzで作成する場合
    np.savez(outfile, x=x, y=y)
    print("npzファイルを保存しました :" + outfile, len(x))

if __name__ == '__main__':
    read_image()