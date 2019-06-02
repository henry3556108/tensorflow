from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Dropout, Flatten, Input, AveragePooling2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.datasets import cifar10
import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model


def resnet_block(inputs,num_filters=16,
                  kernel_size=3,strides=1,
                  activation="relu"):
    x = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',
           kernel_initializer='he_normal',kernel_regularizer="l2")(inputs)
    x = BatchNormalization()(x)
    if(activation):
        x = Activation("relu")(x)
    return x

def rs_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = resnet_block(inputs) # 32,32,16
    for i in range(2):
        a = resnet_block(x) # 32,32,16
        b = resnet_block(a,activation=None) # 32,32,16
        x = keras.layers.add([x,b])
        x = Activation("relu")(x)
    for i in range(2):
        if i == 0 :
            a = resnet_block(x,strides=2,num_filters=32) # 16,16,32
            x = Conv2D(32,kernel_size=3,strides=2,padding='same',kernel_initializer='he_normal',kernel_regularizer="l2")(x)
        else:
            a = resnet_block(x,num_filters=32) # 16,16,32
        b = resnet_block(a,num_filters=32,activation=None) # 16,16,32
        x = keras.layers.add([x,b])
        x = Activation("relu")(x)
    x = resnet_block(x,strides=2,num_filters=64) # 8,8,64
    x = Activation("relu")(x)
    x = keras.layers.AveragePooling2D(pool_size=2)(x)
    y = Flatten()(x)
    y = Dense(256, activation="relu", kernel_initializer='he_normal')(y)
    outputs = Dense(10,activation='softmax',kernel_initializer='he_normal')(y)
    model = Model(inputs=inputs,outputs=outputs)
    return model



def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train/255
    x_test = x_test/255
    y_train = keras.utils.to_categorical(y_train,10)
    y_test = keras.utils.to_categorical(y_test,10)
    input_shape = (32,32,3)
    model = rs_cnn(input_shape)
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr = 0.01,decay=1e-6),metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='rs_model_save/weights-{epoch:02d}.hdf5', verbose=1, save_best_only=True)
    # model.fit(x_train,y_train,batch_size=50,epochs=1,validation_data=(x_test,y_test),verbose=1,callbacks=[checkpointer])
    # scores = model.evaluate(x_test,y_test,verbose=1)
    # print('Test loss:',scores[0])
    # print('Test accuracy:',scores[1])
    plot_model(model, to_file='model.png')

main()



# 使用tensorboard 使用checkpoint