import tensorflow as tf


input_node = 784 #這個是每張mnist數字的像素 28*28
output_node = 10 #我們要將圖片分類成0~9 總共十項
layer1 = 500 #這個是隱藏層的節點 總共有500個

def get_weight(shape,regularizer):
    weight = tf.get_variable("weight",shape,initializer = tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weight)) # 將l2正規化損失加入進losses這個集合裡面
    return weight

def interface(input_tensor,regularizer):
    with tf.variable_scope('lay1'):
        weight = get_weight([input_node,layer1],regularizer)
        biases = tf.get_variable("biases",[layer1],initializer=tf.constant_initializer(0.0)) # 對於這個constant_initializer 0.0有疑問 諞權值全部設為1沒關西嗎
        lay1 = tf.nn.leaky_relu(tf.matmul(input_tensor,weight)+biases) 
    with tf.variable_scope('lay2'):# 不太懂甚麼事variable_scope
        weight = get_weight([layer1,output_node],regularizer)
        biases = tf.get_variable("biases",[output_node],initializer=tf.constant_initializer(0.0)) # 對於這個constant_initializer 0.0有疑問 諞權值全部設為1沒關西嗎
        lay2 = tf.matmul(lay1,weight)+biases
    return lay2