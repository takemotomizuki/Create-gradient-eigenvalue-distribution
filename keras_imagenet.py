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

size = 100
eps = 0.005
img_max = 255.0
#ヒストグラム調整用
adj = 500

#自作したnpzファイル画像サイズは224,224で保存している
d= np.load('../dataset/imagenet_val_float.npz')
x = d['x'][:size]
y = d['y'][:size]

#vgg16
vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet')
x_vgg16 = copy.deepcopy(x)
x_vgg16 = tf.keras.applications.vgg16.preprocess_input(x_vgg16)
preds_vgg16 = np.argmax(vgg16.predict(x_vgg16), axis=1)
acc_vgg16 = np.sum(preds_vgg16 == np.argmax(y, axis=1)) / y.shape[0]
print(acc_vgg16*100,"vgg16 clean img")
grad_vgg16 = get_loss_gradient(vgg16, x, y)
X_adv_vgg16 = copy.deepcopy(x)
X_adv_vgg16 = X_adv_vgg16 + eps*np.sign(grad_vgg16)*img_max
X_adv_vgg16 = np.clip(X_adv_vgg16, 0, 255)
X_adv_vgg16 = tf.keras.applications.vgg16.preprocess_input(X_adv_vgg16)
preds_vgg16_adv = np.argmax(vgg16.predict(X_adv_vgg16), axis=1)
acc_vgg16_adv = np.sum(preds_vgg16_adv == np.argmax(y, axis=1)) / y.shape[0]
print(acc_vgg16_adv*100, "vgg16 adv img")
U, s_vgg16, V = np.linalg.svd(tf.transpose(grad_vgg16), full_matrices=False)

fig, ax = plt.subplots(1, 3,figsize=(12.0, 5.0))
ax[0].hist(s_vgg16[0].reshape(-1)*adj*size,range=(-0.01,5),bins=size,density=True)
ax[0].set_title("vgg16 R")
ax[1].hist(s_vgg16[1].reshape(-1)*adj*size,range=(-0.01,5),bins=size,density=True)
ax[1].set_title("vgg16 G")
ax[2].hist(s_vgg16[2].reshape(-1)*adj*size,range=(-0.01,5),bins=size,density=True)
ax[2].set_title("vgg16 B")
fig.suptitle("eps :" + str(eps) + " clean acc :" + str(acc_vgg16*100) + "%"+ " adv acc :" + str(acc_vgg16_adv*100) + "%")
plt.subplots_adjust(left=0.05, right=0.99)
plt.savefig("assets/vgg16_gradient" + "_size" + str(size) + ".png")
plt.close()

#vgg19
vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet')
x_vgg19 = copy.deepcopy(x)
x_vgg19 = tf.keras.applications.vgg19.preprocess_input(x_vgg19)
preds_vgg19 = np.argmax(vgg19.predict(x_vgg19), axis=1)
acc_vgg19 = np.sum(preds_vgg19 == np.argmax(y, axis=1)) / y.shape[0]
print(acc_vgg19*100,"vgg19 clean img")
grad_vgg19 = get_loss_gradient(vgg19, x, y)
X_adv_vgg19 = copy.deepcopy(x)
X_adv_vgg19 = X_adv_vgg19 + eps*np.sign(grad_vgg19)*img_max
X_adv_vgg19 = np.clip(X_adv_vgg19, 0, 255)
X_adv_vgg19 = tf.keras.applications.vgg19.preprocess_input(X_adv_vgg19)
preds_vgg19_adv = np.argmax(vgg19.predict(X_adv_vgg19), axis=1)
acc_vgg19_adv = np.sum(preds_vgg19_adv == np.argmax(y, axis=1)) / y.shape[0]
print(acc_vgg19_adv*100, "vgg19 adv img")
U, s_vgg19, V = np.linalg.svd(tf.transpose(grad_vgg19), full_matrices=False)

fig, ax = plt.subplots(1, 3,figsize=(12.0, 5.0))
ax[0].hist(s_vgg19[0].reshape(-1)*adj*size,range=(-0.01,5),bins=size,density=True)
ax[0].set_title("vgg19 R")
ax[1].hist(s_vgg19[1].reshape(-1)*adj*size,range=(-0.01,5),bins=size,density=True)
ax[1].set_title("vgg19 G")
ax[2].hist(s_vgg19[2].reshape(-1)*adj*size,range=(-0.01,5),bins=size,density=True)
ax[2].set_title("vgg19 B")
fig.suptitle("eps :" + str(eps) + " clean acc :" + str(acc_vgg19*100) + "%"+ " adv acc :" + str(acc_vgg19_adv*100) + "%")
plt.subplots_adjust(left=0.05, right=0.99)
plt.savefig("assets/vgg19_gradient" + "_size" + str(size) + ".png")
plt.close()

#resnet50
resnet50 = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
x_resnet50 = copy.deepcopy(x)
x_resnet50 = tf.keras.applications.resnet50.preprocess_input(x_resnet50)
preds_resnet50 = np.argmax(resnet50.predict(x_resnet50), axis=1)
acc_resnet50 = np.sum(preds_resnet50 == np.argmax(y, axis=1)) / y.shape[0]
print(acc_resnet50*100,"resnet50 clean img")
grad_resnet50 = get_loss_gradient(resnet50, x, y)
X_adv_resnet50 = copy.deepcopy(x)
X_adv_resnet50 = X_adv_resnet50 + eps*np.sign(grad_resnet50)*img_max
X_adv_resnet50 = np.clip(X_adv_resnet50, 0, 255)
X_adv_resnet50 = tf.keras.applications.resnet50.preprocess_input(X_adv_resnet50)
preds_resnet50_adv = np.argmax(resnet50.predict(X_adv_resnet50), axis=1)
acc_resnet50_adv = np.sum(preds_resnet50_adv == np.argmax(y, axis=1)) / y.shape[0]
print(acc_resnet50_adv*100, "resnet50 adv img")
U, s_resnet50, V = np.linalg.svd(tf.transpose(grad_resnet50), full_matrices=False)

fig, ax = plt.subplots(1, 3,figsize=(12.0, 5.0))
ax[0].hist(s_resnet50[0].reshape(-1)*adj*size,range=(-0.01,5),bins=size,density=True)
ax[0].set_title("resnet50 R")
ax[1].hist(s_resnet50[1].reshape(-1)*adj*size,range=(-0.01,5),bins=size,density=True)
ax[1].set_title("resnet50 G")
ax[2].hist(s_resnet50[2].reshape(-1)*adj*size,range=(-0.01,5),bins=size,density=True)
ax[2].set_title("resnet50 B")
fig.suptitle("eps :" + str(eps) + " clean acc :" + str(acc_resnet50*100) + "%"+ " adv acc :" + str(acc_resnet50_adv*100) + "%")
plt.subplots_adjust(left=0.05, right=0.99)
plt.savefig("assets/resnet50_gradient" + "_size" + str(size) + ".png")
plt.close()