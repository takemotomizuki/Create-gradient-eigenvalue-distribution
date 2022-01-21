from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils

start_layer = 11
end_layer = 20

#mnistデータをロード
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#X_trainの形状を(60000, 28, 28)から(60000, 28, 28, 1)に変更
X_train = X_train.reshape((60000, 28, 28, 1))
#X_testの形状を(10000, 28, 28)から(10000, 28, 28, 1)に変更
X_test = X_test.reshape((10000, 28, 28, 1))

#データの型をfloat32に変える
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#データの正規化(0から255までのデータを0から1までのデータにする)
X_train = X_train / 255
X_test = X_test / 255

#one-hotエンコーディング
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

for ii in range(start_layer,end_layer+1):
    model = Sequential()
    for i in range(ii):
        model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    for i in range(ii):
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.summary()

    #モデルのコンパイル
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics='accuracy')

    #モデルの訓練
    model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2)

    print('\n<Evaluate>')
    score = model.evaluate(X_test, y_test)

    print("test loss score : ", score[0])
    print("test accuracy : ", score[1])

    # save model
    model.save(f"models/mnist_cnn/layer/model_mnist_cnn_{ii}layer.h5")