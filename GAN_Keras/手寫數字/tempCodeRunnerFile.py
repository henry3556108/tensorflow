
from keras.layers.core import Dense, Dropout, Activation 
from keras.optimizers import SGD 
from keras.datasets import mnist 
import numpy 

model = Sequential()
model.add(Dense(500,input_shape=(784,))) # 輸入層，28*28=784 
model.add(Activation('tanh')) # 激活函數是tanh 
model.add(Dropout(0.5)) # 採用50%的dropout 
model.add(Dense(500)) # 隱藏層節點500個 
model.add(Activation('tanh')) 
model.add(Dropout(0.5)) 
model.add(Dense(10)) # 輸出結果是10個類別，所以維度是10 
model.add(Activation('softmax')) # 最後一層用softmax作為激活函數 ''' 第三步：編譯 ''' 
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # 優化函數，設定學習率（lr）等參數 
model.compile(loss='categorical_crossentropy', optimizer=sgd) # 使用交叉熵作為loss函數 ''' 第四步：訓練 .fit的一些參數 batch_size：對總的樣本數進行分組，每組包含的樣本數量 epochs ：訓練次數 shuffle：是否把數據隨機打亂之後再進行訓練 validation_split：拿出百分之多少用來做交叉驗證 verbose：屏顯模式 0：不輸出 1：輸出進度 2：輸出每次的訓練結果 ''' 
(X_train, y_train), (X_test, y_test) = mnist.load_data() # 使用Keras自帶的mnist工具讀取數據（第一次需要聯網） # 由於mist的輸入數據維度是(num, 28, 28)，這裡需要把後面的維度直接拼起來變成784維 
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]) 
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]) 
Y_train = (numpy.arange(10) == y_train[:, None]).astype(int) 
Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)
# print(X_test) 
model.fit(X_train,Y_train,batch_size=200,epochs=50,shuffle=True,verbose=0,validation_split=0.3) 
# model.evaluate(X_test, Y_test, batch_size=200, verbose=0)
print("test set") 
scores = model.evaluate(X_test,Y_test,batch_size=200,verbose=0) 
print("") 
print("The test loss is %f" % scores) 
result = model.predict(X_test,batch_size=200,verbose=0) 
result_max = numpy.argmax(result, axis = 1) 
test_max = numpy.argmax(Y_test, axis = 1) 
result_bool = numpy.equal(result_max, test_max) 
true_num = numpy.sum(result_bool) 
print("") 
print("The accuracy of the model is %f" % (true_num/len(result_bool)))

