import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from numpy.random import RandomState
import numpy as np
import tensorflow as tf


#傳入你的形狀與衰減率生成權重 回傳權重
def get_weight(shape,lamb):
    var = tf.Variable(tf.random_normal(shape),tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(lamb)(var))
    return var
#每次訓練資料大小
batch_size=8
#輸入與輸出資料
_X= tf.placeholder(tf.float32,shape=[None,2])
_Y= tf.placeholder(tf.float32,shape=[None,1])
#待會要計算的資料(將X丟進來)
cur_layer = _X
#每一層的節點量
layer_dim = [2,10,10,10,1]
#我有幾層神經網路
layerlen = len(layer_dim)
#第一層的輸入形狀
in_dimension = layer_dim[0]

for i in range(1 , layerlen):
    #第一層的輸出形狀
    out_dimension = layer_dim[i]
    #呼叫權重
    weight=get_weight([in_dimension,out_dimension],0.0001)
    #你的閥值
    bias = tf.Variable(tf.constant(0.1,shape=[out_dimension]))
    #更新你計算的資料(使用relu)
    cur_layer = tf.nn.relu(tf.matmul(cur_layer,weight)+bias)
    #第一層輸出的形狀就是第二層輸入形狀(要更新)
    in_dimension = out_dimension
#我的損失函數
cross_entropy = -(tf.reduce_mean(_Y * tf.log(tf.clip_by_value(cur_layer,1e-10,1.0))))
tf.add_to_collection('losses',cross_entropy)
loss = tf.add_n(tf.get_collection('losses'))
