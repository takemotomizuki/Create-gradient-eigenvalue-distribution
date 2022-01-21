from numpy.core.fromnumeric import shape
import tensorflow as tf
import ssl

from fgsm_gradient_watcher import FGSM_Gradient_Watcher

ssl._create_default_https_context = ssl._create_unverified_context

def load_model_list(x,y):
    efficientnetB0 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.efficientnet.EfficientNetB0(weights='imagenet',input_shape=(224,224,3)),
        input=x,
        label=y,
        model_name='efficientnetB0',
        )

    efficientnetB1 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.efficientnet.EfficientNetB1(weights='imagenet'),
        input=x,
        label=y,
        model_name='efficientnetB1',
        image_shape=(240,240),
        )

    efficientnetB2 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.efficientnet.EfficientNetB2(weights='imagenet'),
        input=x,
        label=y,
        model_name='efficientnetB2',
        image_shape=(260,260),
        )

    efficientnetB3 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.efficientnet.EfficientNetB3(weights='imagenet'),
        input=x,
        label=y,
        model_name='efficientnetB3',
        image_shape=(300,300),
        )

    efficientnetB4 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.efficientnet.EfficientNetB4(weights='imagenet'),
        input=x,
        label=y,
        model_name='efficientnetB4',
        image_shape=(380,380),
        )

    efficientnetB5 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.efficientnet.EfficientNetB5(weights='imagenet'),
        input=x,
        label=y,
        model_name='efficientnetB5',
        image_shape=(456,456),
        )
    
    efficientnetB6 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.efficientnet.EfficientNetB6(weights='imagenet'),
        input=x,
        label=y,
        model_name='efficientnetB6',
        image_shape=(528,528),
        )

    efficientnetB7 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.efficientnet.EfficientNetB7(weights='imagenet'),
        input=x,
        label=y,
        model_name='efficientnetB7',
        image_shape=(600,600),
        )

    efficientnet_v2B0 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.efficientnet_v2.EfficientNetV2B0(weights='imagenet'),
        input=x,
        label=y,
        model_name='efficientnet_v2B0',
        )

    efficientnet_v2B1 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.efficientnet_v2.EfficientNetV2B1(weights='imagenet'),
        input=x,
        label=y,
        model_name='efficientnet_v2B1',
        image_shape=(240,240),
        )

    efficientnet_v2B2 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.efficientnet_v2.EfficientNetV2B2(weights='imagenet'),
        input=x,
        label=y,
        model_name='efficientnet_v2B2',
        image_shape=(260,260),
        )

    efficientnet_v2B3 = FGSM_Gradient_Watcher(
        model=tf.keras.applications.efficientnet_v2.EfficientNetV2B3(weights='imagenet'),
        input=x,
        label=y,
        model_name='efficientnet_v2B3',
        image_shape=(300,300),
        )

    efficientnet_v2L = FGSM_Gradient_Watcher(
        model=tf.keras.applications.efficientnet_v2.EfficientNetV2L(weights='imagenet'),
        input=x,
        label=y,
        model_name='efficientnet_v2L',
        image_shape=(480,480),
        )

    efficientnet_v2M = FGSM_Gradient_Watcher(
        model=tf.keras.applications.efficientnet_v2.EfficientNetV2M(weights='imagenet'),
        input=x,
        label=y,
        model_name='efficientnet_v2M',
        image_shape=(480,480),
        )

    efficientnet_v2S = FGSM_Gradient_Watcher(
        model=tf.keras.applications.efficientnet_v2.EfficientNetV2S(weights='imagenet'),
        input=x,
        label=y,
        model_name='efficientnet_v2S',
        image_shape=(384,384),
        )
    
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

    mobilenet_v3_large = FGSM_Gradient_Watcher(
        model=tf.keras.applications.MobileNetV3Large(weights='imagenet'),
        input=x,
        label=y,
        model_name='mobilenet_v3_large',
        )

    mobilenet_v3_small = FGSM_Gradient_Watcher(
        model=tf.keras.applications.MobileNetV3Small(weights='imagenet'),
        input=x,
        label=y,
        model_name='mobilenet_v3_small',
        )

    model_list = [
        efficientnetB0,
        efficientnetB1,
        efficientnetB2,
        efficientnetB3,
        efficientnetB4,
        efficientnetB5,
        efficientnetB6,
        efficientnetB7,
        efficientnet_v2B0,
        efficientnet_v2B1,
        efficientnet_v2B2,
        efficientnet_v2B3,
        efficientnet_v2L,
        efficientnet_v2M,
        efficientnet_v2S,
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
        mobilenet_v3_large,
        mobilenet_v3_small,
    ]

    return model_list