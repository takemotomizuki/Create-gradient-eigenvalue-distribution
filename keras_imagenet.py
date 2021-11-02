from os import path
import tensorflow as tf
import numpy as np
import ssl
import logging
import matplotlib.pyplot as plt
import copy


# Configure a logger to capture outputs; these are printed in console and the level of detail is set to INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

loss_object = tf.keras.losses.CategoricalCrossentropy()
ssl._create_default_https_context = ssl._create_unverified_context

tf.enable_eager_execution()

#勾配とそこから作成される敵対画像を作成
def get_loss_gradient(model, _x, _y):
    input_images = tf.multiply(_x, 1)
    input_labels = tf.multiply(_y, 1)
    with tf.GradientTape() as tape:
        tape.watch(input_images)
        prediction = model(input_images)
        loss = loss_object(input_labels, prediction)

        # Get the gradients of the loss w.r.t to the input image.
        gradient = tape.gradient(loss, input_images)
    
    return gradient


def create_eige_ans(_grad_list):
    _s_ans = None
    for g in _grad_list:
        U, s, V = np.linalg.svd(g.T, full_matrices=False)
        if _s_ans is None:
            _s_ans = s
        else:
            np.concatenate([_s_ans,s])
    
    return _s_ans


size = 100
#自作したnpzファイル画像サイズは224,224で保存している
d= np.load('../dataset/imagenet_val_float.npz')
x = d['x'][:size]
y = d['y'][:size]

eps = 0.01
img_max = 255.0

vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet')
x_vgg16 = copy.deepcopy(x)
x_vgg16 = tf.keras.applications.vgg16.preprocess_input(x_vgg16)
preds_vgg16 = np.argmax(vgg16.predict(x_vgg16), axis=1)
acc_vgg16 = np.sum(preds_vgg16 == np.argmax(y, axis=1)) / y.shape[0]
print(acc_vgg16*100,"vgg16 clean img")
grad_vgg16 = get_loss_gradient(vgg16, x, y)
X_adv = copy.deepcopy(x)
X_adv = X_adv + eps*np.sign(grad_vgg16)*img_max
X_adv = np.clip(X_adv, 0, 255)
X_adv = tf.keras.applications.vgg16.preprocess_input(X_adv)
preds_vgg16_adv = np.argmax(vgg16.predict(X_adv), axis=1)
acc_vgg16_adv = np.sum(preds_vgg16_adv == np.argmax(y, axis=1)) / y.shape[0]
print(acc_vgg16_adv*100, "vgg16 adv img")
"""
vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet')
x_vgg19 = copy.deepcopy(x)
x_vgg19 = tf.keras.applications.vgg19.preprocess_input(x_vgg19)
preds_vgg19 = np.argmax(vgg19.predict(x_vgg19), axis=1)
acc_vgg19 = np.sum(preds_vgg19 == np.argmax(y, axis=1)) / y.shape[0]
print(acc_vgg19*100,"vgg19")

resnet50 = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
x_resnet50 = copy.deepcopy(x)
x_resnet50 = tf.keras.applications.resnet50.preprocess_input(x_resnet50)
preds_resnet50 = np.argmax(resnet50.predict(x_resnet50), axis=1)
acc_resnet50 = np.sum(preds_resnet50 == np.argmax(y, axis=1)) / y.shape[0]
print(acc_resnet50*100,"resnet50")
"""

"""
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
"""
