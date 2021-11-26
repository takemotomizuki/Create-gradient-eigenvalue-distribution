import logging

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from fgsm_gradient_watcher import FGSM_Gradient_Watcher
from model_list import load_model_list

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
tf.compat.v1.enable_eager_execution()

size = 300
noise_str = 0.4
#自作したnpzファイル画像サイズは224,224で保存している
d= np.load('../dataset/imagenet_val_float.npz')
x = d['x'][:size]
y = d['y'][:size]

model_list = load_model_list(x,y)

parameter = []

for model in model_list:
    #clean画像の予測精度
    preds = np.argmax(model.predict(model.input), axis=1)
    acc = np.sum(preds == np.argmax(model.label, axis=1)) / len(model.label)
    logger.info(f'{model.model_name} Accuracy on clean test images: {acc*100:.2f}%')

    # 予測を正解ラベルとした攻撃
    preds_onehot = np.eye(1000)[preds]
    grad_preds_base = model.get_loss_gradient(preds_onehot)
    X_adv_preds_base = model.generate_adversarial_image(grad_preds_base,noise_str)
    preds_adv_preds_base = np.argmax(model.predict(X_adv_preds_base), axis=1)
    fooling_rate_preds_base = np.sum(preds != preds_adv_preds_base) / len(preds_adv_preds_base)
    logger.info(f'{model.model_name} Fooling rate on adversarial (preds base) images: {fooling_rate_preds_base*100:.2f}%')
    tale_rate_preds_base = model.calculate_tale_rate(grad_preds_base)

    #正解ラベルに基づいた攻撃
    grad = model.get_loss_gradient()
    X_adv = model.generate_adversarial_image(grad,noise_str)
    preds_adv = np.argmax(model.predict(X_adv), axis=1)
    acc_adv = np.sum(preds_adv == np.argmax(model.label, axis=1)) / len(model.label)
    logger.info(f'{model.model_name} Accuracy on adversarial (label base) images: {acc_adv*100:.2f}%')
    tale_rate = model.calculate_tale_rate(grad)
    
    parameter.append([acc,acc_adv,tale_rate,tale_rate_preds_base,fooling_rate_preds_base])

parameter = np.array(parameter)
cprrcoef = np.corrcoef(parameter[:,3],parameter[:,4])[0,1]

plt.scatter(parameter[:,3],parameter[:,4])
plt.xlabel('Tale rate')
plt.ylabel('Fooling rate (preds base attack)')
plt.title(f'cprrcoef: {cprrcoef}')
plt.subplots_adjust(left=0.05, right=0.99)
plt.savefig(f'assets/Foolingrate_talerate_corrcoef_size{size}.png')
plt.close()
