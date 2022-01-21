from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten
from keras.datasets import cifar10
from keras.utils import np_utils


#cifar10をダウンロード
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

#画像を0-1の範囲で正規化
x_train=x_train.astype('float32')/255.0
x_test=x_test.astype('float32')/255.0

#正解ラベルをOne-Hot表現に変換
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)

for i in range(2,11):
    #モデルを構築
    model=Sequential()

    for ii in range(i):
        model.add(Conv2D(32,(3,3),padding='same',input_shape=(32,32,3)))
        model.add(Activation('relu'))

    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    for ii in range(i):
        model.add(Conv2D(64,(3,3),padding='same'))
        model.add(Activation('relu'))
    
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    history=model.fit(x_train,y_train,batch_size=128,epochs=20,verbose=1,validation_split=0.1)

    #モデルと重みを保存
    json_string=model.to_json()
    open(f'cifar10_cnn{i}.json',"w").write(json_string)
    model.save_weights(f'cifar10_cnn{i}.h5')

    #モデルの表示
    model.summary()

    #評価
    score=model.evaluate(x_test,y_test,verbose=0)
    print('Test loss:',score[0])
    print('Test accuracy:',score[1])
