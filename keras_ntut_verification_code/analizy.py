import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, Conv1D
from keras.layers import MaxPooling2D, MaxPooling1D
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from keras.optimizers import SGD
import cv2

shape = (40, 40)
class_num = 26

np.set_printoptions(threshold=np.inf)


def Build(model):
    model.add(Conv1D(40, 3, input_shape=shape, padding='same'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv1D(40, 3, input_shape=shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(80, 3, input_shape=shape, padding='same'))
    model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(class_num))
    model.add(Activation('softmax'))
    return model


def read_data():
    data = pd.read_csv('./path.csv', index_col=0)
    dic = {}
    for i in range(65, 91):
        dic[chr(i)] = int(i-65)
    data['target'] = pd.DataFrame(map(lambda x: dic[x], data['target']))
    df = shuffle(data)
    return df['path'], df['target']

def read_image(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im //= 255
    return im

def read_data_test():
    data = pd.read_csv('path.csv', index_col=0)
    # 打亂 Dataset 的順序
    data = data.sample(frac=1)
    feature = [read_image(path) for path in data['path'].values]
    feature = np.asarray(feature)
    target = data['target'].values
    label = [ord(x)-65 for x in target]
    label = np.asarray(label)
    return feature, label

def train_test(model, feature, label):
    train_feature, test_feature, train_label, test_label = train_test_split(feature, label, test_size=0.2)

    # train_feature //= 255
    # test_feature //= 255
    # print(train_feature[0])

    train_label = keras.utils.to_categorical(train_label, class_num)
    test_label = keras.utils.to_categorical(test_label, class_num)
    # print(train_label)
    # train_label = keras.preprocessing.text.one_hot(train_label, class_num)
    # test_label = keras.preprocessing.text.one_hot(test_label, class_num)
    # print(train_label)
    # pass
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_feature, train_label, batch_size=300, epochs=5, verbose=1)
    result = model.evaluate(test_feature, test_label, verbose=1)
    print('Test lost:', result[0], ', accuracy:', result[1])
    return

def Train(model, feature_path, target):
    ls = list(map(lambda x: cv2.imread(x), feature_path))
    # print(ls[0].shape)
    ls = np.array(ls)
    print(target.shape)
    train_feature, test_feature, train_label, test_label = train_test_split(
        ls, np.array(target), test_size=0.2)
    sgd = SGD(lr=0.3, decay=1e-6, momentum=0.9,
              nesterov=True)  # 優化函數，設定學習率（lr）等參數
    train_label = keras.utils.to_categorical(train_label)
    test_label = keras.utils.to_categorical(test_label)
    # # print(test_label[0])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_feature, train_label, batch_size=50, epochs=5, verbose=1)
    result = model.predict(test_feature)
    result_bool = np.equal(result, test_label)
    true_num = np.sum(result_bool)
    print("")
    print("The accuracy of the model is %f" % (true_num/len(result_bool)))


def main():
    # path, target = read_data()
    # print(path,target)
    feature, label = read_data_test()
    # print(label)
    model = Sequential()
    model = Build(model)
    # Train(model, path, target)
    train_test(model, feature, label)


if __name__ == '__main__':
    main()
