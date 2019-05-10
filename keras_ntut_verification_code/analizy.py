from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras.optimizers import SGD
import cv2

shape = (40,40,3)


def Build(model):
    # 第一個部分的conv跟pool
    model.add(Conv2D(32, (3, 3), input_shape=shape, padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(Conv2D(32, (3, 3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(Conv2D(32, (3, 3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (3, 3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(Conv2D(128, (3, 3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(Conv2D(128, (3, 3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, (3, 3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(Conv2D(256, (3, 3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(Conv2D(256, (3, 3), padding='same',activation='relu',kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu',kernel_initializer='he_normal'))
    model.add(Dense(128, activation='relu',kernel_initializer='he_normal'))
    model.add(Dense(26, activation='softmax'))
    return model

def read_data():
    data = pd.read_csv('path.csv',index_col=0)
    dic = {}
    for i in range(65,91):
        dic[chr(i)] = int(i-65)
    data['target'] = pd.DataFrame(map(lambda x : dic[x],data['target']))
    df = shuffle(data)
    return df['path'],df['target']

def Train(model,feature_path,target):
    ls = list(map(lambda x : cv2.imread(x),feature_path))
    # print(ls[0].shape)
    ls = np.array(ls)
    print(target.shape)
    train_feature,test_feature,train_label,test_label = train_test_split(ls,np.array(target),test_size = 0.2)
    sgd = SGD(lr=0.3, decay=1e-6, momentum=0.9, nesterov=True) # 優化函數，設定學習率（lr）等參數
    train_label = keras.utils.to_categorical(train_label)
    test_label = keras.utils.to_categorical(test_label)
    # # print(test_label[0])
    model.compile(loss='categorical_crossentropy',optimizer='adam')
    model.fit(train_feature,train_label, batch_size=50, epochs=5, verbose=1)
    result = model.predict(test_feature)
    result_bool = np.equal(result, test_label) 
    true_num = np.sum(result_bool) 
    print("") 
    print("The accuracy of the model is %f" % (true_num/len(result_bool)))
    



def main():
    path,target = read_data()
    # print(path,target)
    model = Sequential()
    model=Build(model)
    Train(model,path,target)
if __name__ == '__main__':
    main()