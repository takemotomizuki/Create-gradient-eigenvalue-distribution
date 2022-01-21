import logging
from operator import index

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

import powerlaw

from fgsm_gradient_watcher import FGSM_Gradient_Watcher
from model_list_cnn_mnist import load_model_list

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

size = 10000
noise_str = 0.2

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
x_train, x_test = x_train / 255.0, x_test / 255.0

# テスト画像からいくつか選択
x = x_test[:size]
y = np.eye(10)[y_test[:size]]

model_list = load_model_list(x,y)

parameter = []

for model in model_list:
    #clean画像の予測精度
    preds = np.argmax(model.predict(model.input), axis=1)
    acc = np.sum(preds == np.argmax(model.label, axis=1)) / len(model.label)
    logger.info(f'{model.model_name} Accuracy on clean test images: {acc*100:.2f}%')

    #正解ラベルに基づいた攻撃
    grad = model.get_loss_gradient()
    X_adv = model.generate_adversarial_image(grad,noise_str)
    preds_adv = np.argmax(model.predict(X_adv), axis=1)
    acc_adv = np.sum(preds_adv == np.argmax(model.label, axis=1)) / len(model.label)
    logger.info(f'{model.model_name} Accuracy on adversarial (label base) images: {acc_adv*100:.2f}%')
    tale_rate = model.calculate_tale_rate(grad)

    U, s, V = np.linalg.svd(tf.transpose(grad), full_matrices=False)
    xmax=np.max(s)
    xmin=xmax/100
    fit = powerlaw.Fit(s.reshape(-1), xmax=xmax, xmin=xmin, verbose=False)
    alpha = fit.alpha

    # 予測を正解ラベルとした攻撃
    preds_onehot = np.eye(10)[preds]
    grad_preds_base = model.get_loss_gradient(preds_onehot)
    X_adv_preds_base = model.generate_adversarial_image(grad_preds_base,noise_str)
    preds_adv_preds_base = np.argmax(model.predict(X_adv_preds_base), axis=1)
    fooling_rate_preds_base = np.sum(preds != preds_adv_preds_base) / len(preds_adv_preds_base)
    logger.info(f'{model.model_name} Fooling rate on adversarial (preds base) images: {fooling_rate_preds_base*100:.2f}%')
    tale_rate_preds_base = model.calculate_tale_rate(grad_preds_base)

    U, s, V = np.linalg.svd(tf.transpose(grad_preds_base), full_matrices=False)
    xmax=np.max(s)
    xmin=xmax/100
    fit = powerlaw.Fit(s.reshape(-1), xmax=xmax, xmin=xmin, verbose=False)
    alpha_preds_base = fit.alpha
    
    parameter.append([
        model.model_name,
        acc,
        acc_adv,
        tale_rate,
        alpha,
        tale_rate_preds_base,
        fooling_rate_preds_base,
        alpha_preds_base
        ])

df = pd.DataFrame(
  parameter,
  columns=[
    'model_name', 
    'clean_acc', 
    'adv_acc_label_base', 
    'tale_rate_label_base', 
    'alpha_label_base',
    'talerate_preds_base', 
    'foolingrate_preds_rate',
    'alpha_preds_base',
    ])

df.to_csv(f'assets/parameter_cnn_mnist_layer_size{size}.csv')