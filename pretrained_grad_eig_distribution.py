import tensorflow as tf
import numpy as np
import ssl
import logging
import matplotlib.pyplot as plt
import copy
from PIL import Image
import powerlaw
import math

# Configure a logger to capture outputs; these are printed in console and the level of detail is set to INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

loss_object = tf.keras.losses.CategoricalCrossentropy()
ssl._create_default_https_context = ssl._create_unverified_context

tf.compat.v1.enable_eager_execution()

size = 100
eps = 0.003

def np_extend(ndarray:np.ndarray,range_min:int,range_max:int):
    """
    numpy配列を任意の範囲に引き伸ばす
    """
    a_max = np.max(ndarray)
    a_min = np.min(ndarray)

    ndarray = (ndarray-a_min)/(a_max-a_min)
    ndarray = ndarray*(range_max-range_min)+range_min

    return ndarray

#勾配とそこから作成される敵対画像を作成
def get_loss_gradient(model, _x, _y):
    num = len(_x)/50
    x_split = np.array_split(_x, num+1)
    y_split = np.array_split(_y, num+1)
    gradient = None
    for xs,ys in zip(x_split,y_split):
        input_images = tf.multiply(xs, 1)
        input_labels = tf.multiply(ys, 1)
        with tf.GradientTape() as tape:
            tape.watch(input_images)
            prediction = model(input_images)
            loss = loss_object(input_labels, prediction)

            # Get the gradients of the loss w.r.t to the input image.
            g = tape.gradient(loss, input_images)
            if gradient is None:
                gradient = g
            else:
                gradient = tf.concat([gradient,g],0)

    return gradient

def calc_tale_size(evals):
    max_eval = np.max(evals)

    evals_sorted = np.sort(evals)
    size = len(evals)
    border_size = int(size*0.8)

    return evals_sorted[border_size]/max_eval

#atk_success rate = acc_adv/acc
def atk_FGSM_and_plot_grad_eig(_x,_y,model,application:function,shape=None,name:str=""):
    x_preprocessed = copy.deepcopy(_x)
    if shape is not None:
        tmp = []
        for img in x_preprocessed:
            img_pil = Image.fromarray(np.uint8(img))
            img_resize = img_pil.resize(shape)
            tmp.append(np.asarray(img_resize))
        x_preprocessed = np.array(tmp)
    x_preprocessed = application.preprocess_input(x_preprocessed)
    x_max = np.max(x_preprocessed)
    x_min = np.min(x_preprocessed)
    preds = np.argmax(model.predict(x_preprocessed), axis=1)
    acc = np.sum(preds == np.argmax(_y, axis=1)) / _y.shape[0]
    grad = get_loss_gradient(model, x_preprocessed, _y)
    X_adv = x_preprocessed + eps*np_extend(np.sign(grad),x_min,x_max)
    preds_adv = np.argmax(model.predict(X_adv), axis=1)
    acc_adv = np.sum(preds_adv == np.argmax(_y, axis=1)) / _y.shape[0]
    
    logger.info(f'Accuracy on clean test images of {name}: {acc*100:.2f}')
    logger.info(f'Accuracy of FGSM attacks of {name}: {acc_adv*100:.2f}%')
    logger.info(f'Attack succes rate: {(1-acc_adv/acc)*100:.2f}%')

    U, s, V = np.linalg.svd(tf.transpose(grad), full_matrices=False)

    tale_size = calc_tale_size(s[0].reshape(-1))
    logger.info(f'Tale rate : {tale_size}')
    
    label = ["R","G","B"]
    range_max = np.max(s)
    fig, ax = plt.subplots(1, 3,figsize=(12.0, 5.0))

    for a,l,ss in zip(ax,label,s):
        a.hist(ss.reshape(-1),range=(0,range_max),bins=100,density=True)
        a.set_title(l)
    fig.suptitle(f'{name} eps : {eps} clean acc : {acc*100:.2f}% FGSM acc : {acc_adv*100:.2f}%')
    plt.subplots_adjust(left=0.05, right=0.99)
    plt.savefig(f'assets/{name}_gradient_size{size}.png')
    plt.close()

   
    #fit = powerlaw.Fit(nz_s, xmax=range_max, verbose=False)

    range_min_log = math.log10(np.min(s))
    range_max_log = math.log10(range_max)

    fig, ax = plt.subplots(1, 3,figsize=(24.0, 5.0))
    for a,l,ss in zip(ax,label,s):
        a.set_xscale("log")
        a.set_yscale("log")
        a.hist(ss.reshape(-1),bins=np.logspace(range_min_log,range_max_log,100),density=True)
        a.set_title(l)

    fig.suptitle(f'{name} eps : {eps} clean acc : {acc*100:.2f}% FGSM acc : {acc_adv*100:.2f}%')
    plt.subplots_adjust(left=0.01, right=0.99)
    plt.savefig(f'assets/{name}_gradient_size{size}_log.png')
    plt.close()
    logger.info('saved image')

    return tale_size, (1-acc_adv/acc)*100
    

#自作したnpzファイル画像サイズは224,224で保存している
d= np.load('../dataset/imagenet_val_float.npz')
x = d['x'][:size]
y = d['y'][:size]

tale_rate_list = []
atk_succes_list = []

#xception
xception = tf.keras.applications.xception.Xception(weights='imagenet')
t,a = atk_FGSM_and_plot_grad_eig(x,y,xception,tf.keras.applications.xception,shape=(299,299),name="xception")
tale_rate_list.append(t)
atk_succes_list.append(a)

#vgg16
vgg16 = tf.keras.applications.vgg16.VGG16(weights='imagenet')
t,a = atk_FGSM_and_plot_grad_eig(x,y,vgg16,tf.keras.applications.vgg16,name="vgg16")
tale_rate_list.append(t)
atk_succes_list.append(a)

#vgg19m
vgg19 = tf.keras.applications.vgg19.VGG19(weights='imagenet')
t,a = atk_FGSM_and_plot_grad_eig(x,y,vgg19,tf.keras.applications.vgg19,name="vgg19")
tale_rate_list.append(t)
atk_succes_list.append(a)

#resnet50
resnet50 = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
t,a = atk_FGSM_and_plot_grad_eig(x,y,resnet50,tf.keras.applications.resnet50,name="resnet50")
tale_rate_list.append(t)
atk_succes_list.append(a)

#inception_v3
inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')
t,a = atk_FGSM_and_plot_grad_eig(x,y,inception_v3,tf.keras.applications.inception_v3,shape=(299,299),name="inception_v3")
tale_rate_list.append(t)
atk_succes_list.append(a)

#inception_resnet_v2
inception_resnet_v2 = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet')
t,a = atk_FGSM_and_plot_grad_eig(x,y,inception_resnet_v2,tf.keras.applications.inception_resnet_v2,shape=(299,299),name="inception_resnet_v2")
tale_rate_list.append(t)
atk_succes_list.append(a)

#mobilenet
mobilenet = tf.keras.applications.mobilenet.MobileNet(weights='imagenet')
t,a = atk_FGSM_and_plot_grad_eig(x,y,mobilenet,tf.keras.applications.mobilenet,name="mobilenet")
tale_rate_list.append(t)
atk_succes_list.append(a)

#densenet
densenet121 = tf.keras.applications.densenet.DenseNet121(weights='imagenet')
t,a = atk_FGSM_and_plot_grad_eig(x,y,densenet121,tf.keras.applications.densenet,name="densenet121")
tale_rate_list.append(t)
atk_succes_list.append(a)
densenet169 = tf.keras.applications.densenet.DenseNet169(weights='imagenet')
t,a = atk_FGSM_and_plot_grad_eig(x,y,densenet169,tf.keras.applications.densenet,name="densenet169")
tale_rate_list.append(t)
atk_succes_list.append(a)
densenet201 = tf.keras.applications.densenet.DenseNet201(weights='imagenet')
t,a = atk_FGSM_and_plot_grad_eig(x,y,densenet201,tf.keras.applications.densenet,name="densenet201")
tale_rate_list.append(t)
atk_succes_list.append(a)

#NASNet
nasnet_large = tf.keras.applications.nasnet.NASNetLarge(weights='imagenet')
t,a = atk_FGSM_and_plot_grad_eig(x,y,nasnet_large,tf.keras.applications.nasnet,shape=(331,331),name="nasnet_large")
tale_rate_list.append(t)
atk_succes_list.append(a)
nasnet_mobile = tf.keras.applications.nasnet.NASNetMobile(weights='imagenet')
t,a = atk_FGSM_and_plot_grad_eig(x,y,nasnet_mobile,tf.keras.applications.nasnet,name="nasnet_mobile")
tale_rate_list.append(t)
atk_succes_list.append(a)

#mobilenet_v2
mobilenet_v2 = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet')
t,a = atk_FGSM_and_plot_grad_eig(x,y,mobilenet_v2,tf.keras.applications.mobilenet_v2,name="mobilenet_v2")
tale_rate_list.append(t)
atk_succes_list.append(a)

plt.scatter(tale_rate_list, atk_succes_list)
plt.xlabel("Tale rate")
plt.ylabel("Atk success rate")
plt.savefig(f'assets/tale_rate_size{size}.png')
plt.close()