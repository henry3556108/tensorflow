from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

A = np.zeros(200)
A = A.reshape([2,100])
for index, value in enumerate(np.linspace(start=-1, stop=1, num=100)):
    A[0][index] = value 
    
np.random.shuffle(A[0])
# print(A)    
Y = 5 * A[0]**3 - 6 * A[0]**2 + 4 * A[0] + 2 + np.random.normal(loc=0, scale=1, size=100)
A[1]=A[0]**6
# print(A)
# XX=[]
# XX.append(X)

# # XX=[list(X),list(X**2)]
# print(XX)
# print(len(X),len(X2))
# A = A[::-1]
model = Sequential()
model.add(Dense(10,input_shape=(2,),activation='linear',kernel_initializer='uniform'))
model.add(Dense(1,activation='linear',kernel_initializer='uniform'))
model.compile(loss='mean_squared_error', optimizer='sgd') # 使用交叉熵作為loss函數 ''' 第四步：訓練 .fit的一些參數 batch_size：對總的樣本數進行分組，每組包含的樣本數量 epochs ：訓練次數 shuffle：是否把數據隨機打亂之後再進行訓練 validation_split：拿出百分之多少用來做交叉驗證 verbose：屏顯模式 0：不輸出 1：輸出進度 2：輸出每次的訓練結果 ''' 
model.fit(A.T,Y, batch_size=10, epochs=200, verbose=1)
# result = model.predict([X])
Y_pred = model.predict(A.T)
plt.scatter(A[0], Y)
plt.scatter(A[0], Y_pred)
plt.show()
