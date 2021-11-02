from os import path
import tensorflow as tf
import numpy as np
import ssl
import logging
import matplotlib.pyplot as plt
from pathlib import Path


# Configure a logger to capture outputs; these are printed in console and the level of detail is set to INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

loss_object = tf.keras.losses.CategoricalCrossentropy()
ssl._create_default_https_context = ssl._create_unverified_context
AUTOTUNE = tf.data.experimental.AUTOTUNE

# path 以下の画像を読み込む
all_image_paths = list(Path('../ILSVRC2012_img_val').glob('*.JPEG'))
all_image_paths.sort()
all_image_paths = all_image_paths[:1000]
all_image_paths = [str(path) for path in all_image_paths]

tf.enable_eager_execution()

#前処理の方法がモデルごとに違うのでそれぞれの関数を定義
def load_and_preprocess_image_res50(path):
    image_raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.resnet50.preprocess_input(image)

    return image

def load_and_preprocess_image_vgg16(path):
    image_raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.vgg16.preprocess_input(image)

    return image

def load_and_preprocess_image_vgg19(path):
    image_raw = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_raw, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.vgg19.preprocess_input(image)

    return image

#勾配とそこから作成される敵対画像を作成
def create_adversarial_image(model, _x, _y, eps):
    _X_adv = []
    grad_ans = []
    for n,input_image in enumerate(_x.take(100)):
        input_label = _y[n]
        input_image = input_image[None, ...]
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            prediction = model(input_image)
            loss = loss_object(input_label, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_image)
        signed_grad = tf.sign(gradient)
        _X_adv.append(input_image + eps*signed_grad)
        grad_ans.append(gradient[0].numpy())
    return np.array(grad_ans),_X_adv

def create_eige_ans(_grad_list):
    _s_ans = None
    for g in _grad_list:
        U, s, V = np.linalg.svd(g.T, full_matrices=False)
        if _s_ans is None:
            _s_ans = s
        else:
            np.concatenate([_s_ans,s])
    
    return _s_ans


d= np.load('../dataset/imagenet_val.npz')
y = d['y'][:100]
y = np.eye(1000)[y]

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

image_ds_vgg16 = path_ds.map(load_and_preprocess_image_vgg16, num_parallel_calls=AUTOTUNE)
vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet')

image_ds_vgg19 = path_ds.map(load_and_preprocess_image_vgg19, num_parallel_calls=AUTOTUNE)
vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet')

image_ds_res50 = path_ds.map(load_and_preprocess_image_res50, num_parallel_calls=AUTOTUNE)
resnet50 = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

grad_list = create_adversarial_image(resnet50, image_ds, y, 0.5)
s_ans = create_eige_ans(grad_list)

plt.hist(s_ans[0],range=(-0.001,0.005),bins=100)
plt.savefig('assets/resnet50_r.png')
plt.close()
plt.hist(s_ans[1],range=(-0.001,0.005),bins=100)
plt.savefig('assets/resnet50_g.png')
plt.close()
plt.hist(s_ans[2],range=(-0.001,0.005),bins=100)
plt.savefig('assets/resnet50_b.png')
plt.close()
