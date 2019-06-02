from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.datasets import cifar10
import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.utils import plot_model

input_shape = (32,32,3)

def Build(model):
    # 第一個部分的conv跟pool
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # 第二part的conv pool
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal'))
    # model.add(Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal'))
    # model.add(Conv2D(128, (3, 3), activation='relu', padding='same',kernel_initializer='he_normal'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu',kernel_initializer='he_normal'))
    model.add(Dense(10, activation='softmax'))
    return model

# def read_data():
#     return 1

def Train(model):
    # data = read_data()
    (train_feature,train_label),(test_feature,test_label) = cifar10.load_data()
    train_feature = train_feature/255
    test_feature = test_feature/255
    print(train_feature.shape)
    train_label = keras.utils.to_categorical(train_label, 10)
    test_label = keras.utils.to_categorical(test_label, 10)
    print(len(train_feature))
    model.compile(loss='categorical_crossentropy',optimizer='adam') 
    model.fit(train_feature,train_label, batch_size=200, epochs=1, verbose=1)
    result = model.predict(test_feature)
    result_bool = np.equal(result, test_label) 
    true_num = np.sum(result_bool) 
    print("") 
    print("The accuracy of the model is %f" % (true_num/len(result_bool)))
    



def main():
    model = Sequential()
    model=Build(model)
    Train(model)
    model.save('my_model.h5')
    plot_model(model, to_file='model.png')

if __name__ == '__main__':
    main()