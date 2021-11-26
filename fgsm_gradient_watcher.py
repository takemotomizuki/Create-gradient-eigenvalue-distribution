import logging

import tensorflow as tf
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import math
import copy

logger = logging.getLogger(__name__)

class FGSM_Gradient_Watcher():
    def __init__(
        self,
        model,
        input:np.ndarray,
        label:np.ndarray,
        model_name:str=None,
        image_shape:tuple=None,
        preprocess=None,
        mode:str=None,
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        ):
        self.model:tf.keras.Model = model
        self.input:np.ndarray = input
        self.label:np.ndarray = label
        if image_shape is not None:
            self.input = self.resize_image(image_shape)
        if preprocess is not None:
            self.input = self.preprocess_input(preprocess)
        self.model_name:str = model_name
        self.mode:str = mode
        self.loss_object = loss_object

    def preprocess_input(self,preprocess):
        x_preprocessed = copy.deepcopy(self.input)
        x_preprocessed = preprocess(x_preprocessed)
        return x_preprocessed
    
    def resize_image(self,shape)->np.ndarray:
        tmp = []
        for img in self.input:
            img_pil = Image.fromarray(np.uint8(img))
            img_resize = img_pil.resize(shape)
            tmp.append(np.asarray(img_resize))
        return np.array(tmp)

    def predict(self,x):
        return self.model.predict(x)

    def get_loss_gradient(self,y=None)->np.ndarray:   
        if y is None:
            y=self.label

        num = len(self.input)/50
        x_split = np.array_split(self.input, num+1)
        y_split = np.array_split(y, num+1)
        gradient = None
        for xs,ys in zip(x_split,y_split):
            input_images = tf.multiply(xs, 1)
            input_labels = tf.multiply(ys, 1)
            with tf.GradientTape() as tape:
                tape.watch(input_images)
                prediction = self.model(input_images)
                loss = self.loss_object(input_labels, prediction)

                # Get the gradients of the loss w.r.t to the input image.
                g = tape.gradient(loss, input_images)
                if gradient is None:
                    gradient = g
                else:
                    gradient = tf.concat([gradient,g],0)
        return gradient.numpy()

    def generate_adversarial_image(self,gradient,noise_str)->np.ndarray:
        eps = [noise_str,noise_str,noise_str]
        if self.mode is not None:
            if self.mode == 'tf':
                eps[0] = eps[0]/127.5
                eps[1] = eps[1]/127.5
                eps[2] = eps[2]/127.5
            elif self.mode == 'torch':
                eps[0] = eps[0] / 255.0 / 0.229 # for R
                eps[1] = eps[1] / 255.0 / 0.224 # for G
                eps[2] = eps[2] / 255.0 / 0.225 # for B
    
        x = copy.deepcopy(self.input)
        X_adv = x + eps*np.sign(gradient)
        return X_adv

    def plot_gradient_eval_distribution(self,gradient)->None:
        U, s, V = np.linalg.svd(tf.transpose(gradient), full_matrices=False)

        label = ["R","G","B"]
        range_max = np.max(s)
        fig, ax = plt.subplots(1, 3,figsize=(12.0, 5.0))

        for a,l,ss in zip(ax,label,s):
            a.hist(ss.reshape(-1),range=(0,range_max),bins=100,density=True)
            a.set_title(l)
        fig.suptitle(f'{self.model_name}')
        plt.subplots_adjust(left=0.05, right=0.99)
        plt.savefig(f'assets/{self.model_name}_gradient_size{len(self.input)}.png')
        plt.close()

        range_min_log = math.log10(np.min(s))
        range_max_log = math.log10(range_max)

        fig, ax = plt.subplots(1, 3,figsize=(24.0, 5.0))
        for a,l,ss in zip(ax,label,s):
            a.set_xscale("log")
            a.set_yscale("log")
            a.hist(ss.reshape(-1),bins=np.logspace(range_min_log,range_max_log,100),density=True)
            a.set_title(l)

        fig.suptitle(f'{self.model_name}')
        plt.subplots_adjust(left=0.01, right=0.99)
        plt.savefig(f'assets/{self.model_name}_gradient_size{len(self.input)}_log.png')
        plt.close()
        logger.info('saved image')

    def calculate_tale_rate(self,gradient)->float:
        U, s, V = np.linalg.svd(tf.transpose(gradient), full_matrices=False)
        max_eval = np.max(s)

        evals_sorted = np.sort(s.reshape(-1))
        size = len(s)
        border_size = int(size*0.8)

        return evals_sorted[border_size]/max_eval