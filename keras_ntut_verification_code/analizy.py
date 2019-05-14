from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from sklearn.model_selection import train_test_split

import keras
import pandas as pd
import numpy as np
import cv2

shape = (40, 40)
class_num = 26

# 強制 numpy 顯示完整的陣列
# np.set_printoptions(threshold=np.inf)


def build(model):
    '''建立模型'''
    model.add(Conv1D(40, 3, input_shape=shape, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(40, 3, input_shape=shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(80, 3, input_shape=shape, padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(class_num))
    model.add(Activation('softmax'))
    return model


def read_image(path):
    '''從指定路徑讀取圖片
    path -- 圖片的路徑 (包含副檔名)
    '''
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # 將圖片二值化
    im //= 255
    return im


def read_data(csv: str):
    '''從 csv 檔讀取 feature 跟 label
    csv -- 指定的 csv 檔案路徑 (包含副檔名)
    '''
    data = pd.read_csv(csv, index_col=0)
    # 打亂 Dataset 的順序
    data = data.sample(frac=1)
    feature = [read_image(path) for path in data['path'].values]
    feature = np.asarray(feature)
    target = data['target'].values
    # 將 label 轉為數字
    label = [ord(x)-65 for x in target]
    label = np.asarray(label)
    return feature, label


def train(model, feature, label, epochs):
    '''訓練指定的模型，並進行驗證'''
    # 分割 Dataset
    train_feature, test_feature, train_label, test_label = train_test_split(
        feature, label, test_size=0.2)

    train_label = keras.utils.to_categorical(train_label, class_num)
    test_label = keras.utils.to_categorical(test_label, class_num)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(train_feature, train_label,
              batch_size=300, epochs=epochs, verbose=1)
    result = model.evaluate(test_feature, test_label, verbose=1)
    print('Test lost:', result[0], ', accuracy:', result[1])
    return model


def main():
    feature, label = read_data('path.csv')
    model = Sequential()
    model = build(model)
    train(model, feature, label, epochs=20)


if __name__ == '__main__':
    main()
