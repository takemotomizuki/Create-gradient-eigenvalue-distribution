import tensorflow as tf
import numpy as np

from fgsm_gradient_watcher import FGSM_Gradient_Watcher

tf.compat.v1.enable_eager_execution()

def load_model_list(x,y):
    xception = FGSM_Gradient_Watcher(
        model=tf.keras.applications.xception.Xception(weights='imagenet'),
        input=x,
        label=y,
        model_name='xception',
        image_shape=(299,299),
        preprocess=tf.keras.applications.xception.preprocess_input,
        mode='tf',
        )

    vgg16 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.vgg16.VGG16(weights='imagenet'),
        input=x,
        label=y,
        model_name='vgg16',
        preprocess=tf.keras.applications.vgg16.preprocess_input,
        mode='caffe',
        )

    vgg19 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.vgg19.VGG19(weights='imagenet'),
        input=x,
        label=y,
        model_name='vgg19',
        preprocess=tf.keras.applications.vgg19.preprocess_input,
        mode='caffe',
        )

    resnet50 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.resnet50.ResNet50(weights='imagenet'),
        input=x,
        label=y,
        model_name='resnet50',
        preprocess=tf.keras.applications.resnet50.preprocess_input,
        mode='caffe',
        )
    
    resnet101 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.resnet.ResNet101(weights='imagenet'),
        input=x,
        label=y,
        model_name='resnet101',
        preprocess=tf.keras.applications.resnet50.preprocess_input,
        mode='caffe',
        )

    resnet152 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.resnet.ResNet152(weights='imagenet'),
        input=x,
        label=y,
        model_name='resnet152',
        preprocess=tf.keras.applications.resnet50.preprocess_input,
        mode='caffe',
        )

    resnet50_v2 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.resnet_v2.ResNet50V2(weights='imagenet'),
        input=x,
        label=y,
        model_name='resnet50_v2',
        preprocess=tf.keras.applications.resnet_v2.preprocess_input,
        mode='tf',
        )

    resnet101_v2 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.resnet_v2.ResNet101V2(weights='imagenet'),
        input=x,
        label=y,
        model_name='resnet101_v2',
        preprocess=tf.keras.applications.resnet_v2.preprocess_input,
        mode='tf',
        )

    resnet152_v2 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.resnet_v2.ResNet152V2(weights='imagenet'),
        input=x,
        label=y,
        model_name='resnet152_v2',
        preprocess=tf.keras.applications.resnet_v2.preprocess_input,
        mode='tf',
        )

    inception_v3 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.inception_v3.InceptionV3(weights='imagenet'),
        input=x,
        label=y,
        model_name='inception_v3',
        image_shape=(299,299),
        preprocess=tf.keras.applications.inception_v3.preprocess_input,
        mode='tf',
        )

    inception_resnet_v2 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet'),
        input=x,
        label=y,
        model_name='inception_resnet_v2',
        image_shape=(299,299),
        preprocess=tf.keras.applications.inception_resnet_v2.preprocess_input,
        mode='tf',
        )

    densenet121 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.densenet.DenseNet121(weights='imagenet'),
        input=x,
        label=y,
        model_name='densenet121',
        preprocess=tf.keras.applications.densenet.preprocess_input,
        mode='torch',
        )

    densenet169 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.densenet.DenseNet169(weights='imagenet'),
        input=x,
        label=y,
        model_name='densenet169',
        preprocess=tf.keras.applications.densenet.preprocess_input,
        mode='torch',
        )

    densenet201 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.densenet.DenseNet201(weights='imagenet'),
        input=x,
        label=y,
        model_name='densenet201',
        preprocess=tf.keras.applications.densenet.preprocess_input,
        mode='torch',
        )

    nasnet_large = FGSM_Gradient_Watcher(
        model=tf.keras.applications.nasnet.NASNetLarge(weights='imagenet'),
        input=x,
        label=y,
        model_name='nasnet_large',
        image_shape=(331,331),
        preprocess=tf.keras.applications.nasnet.preprocess_input,
        mode='tf',
        )

    nasnet_mobile = FGSM_Gradient_Watcher(
        model=tf.keras.applications.nasnet.NASNetMobile(weights='imagenet'),
        input=x,
        label=y,
        model_name='nasnet_mobile',
        preprocess=tf.keras.applications.nasnet.preprocess_input,
        mode='tf',
        )

    mobilenet = FGSM_Gradient_Watcher(
        model=tf.keras.applications.mobilenet.MobileNet(weights='imagenet'),
        input=x,
        label=y,
        model_name='mobilenet',
        preprocess=tf.keras.applications.mobilenet.preprocess_input,
        mode='tf',
        )

    mobilenet_v2 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet'),
        input=x,
        label=y,
        model_name='mobilenet_v2',
        preprocess=tf.keras.applications.mobilenet_v2.preprocess_input,
        mode='tf',
        )

    model_list = [
        xception,
        vgg16,
        vgg19,
        resnet50,
        resnet101,
        resnet152,
        resnet50_v2,
        resnet101_v2,
        resnet152_v2,
        inception_v3,
        inception_resnet_v2,
        densenet121,
        densenet169,
        densenet201,
        nasnet_large,
        nasnet_mobile,
        mobilenet,
        mobilenet_v2,
    ]

    return model_list



