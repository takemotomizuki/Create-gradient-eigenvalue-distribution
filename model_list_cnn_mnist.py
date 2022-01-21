from numpy.core.fromnumeric import shape
import tensorflow as tf
import ssl

from fgsm_gradient_watcher import FGSM_Gradient_Watcher

ssl._create_default_https_context = ssl._create_unverified_context

def load_model_list(x,y):
    model_list = []
    for i in range(1,11):
        model = FGSM_Gradient_Watcher(
            model=tf.keras.models.load_model(f'models/mnist_cnn/layer/model_mnist_cnn_{i}layer.h5'),
            input=x,
            label=y,
            model_name=f'model_mnist_cnn_{i}layer',
            mode = 'channel1',
            loss_object=tf.keras.losses.CategoricalCrossentropy()
        )
        model_list.append(model)

    return model_list