import tensorflow as tf
import hashlib as hs
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from numpy import *
mnist = input_data.read_data_sets("C:\\Users\\user\\Documents\\GitHub\\tensorflow",one_hot=True)

print(shape( mnist.test.labels))

# regularization_rate = 0.0001 # 不太清楚
# weight1 = tf.Variable(tf.truncated_normal([100,200],stddev = 0.1))
# regularize = tf.contrib.layers.l2_regularizer(regularization_rate)# 使用L2正規化來計算
# r = regularize(weight1)
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(r))

















# v1 = tf.Variable(0, dtype=tf.float32)
# step = tf.Variable(0, trainable=False)
# ema = tf.train.ExponentialMovingAverage(0.99, step)
# maintain_averages_op = ema.apply([v1])
# with tf.Session() as sess: # 初始化 
#     init_op = tf.global_variables_initializer() 
#     sess.run(init_op) 
#     print (sess.run([v1, ema.average(v1)])) # 更新变量v1的取值，衰减率为min{0.99， 1+step/10+step=0.1}=0.1，所以v1的滑动平均被更新为0.1*0+0.9*5=4.5 
#     sess.run(tf.assign(v1, 5)) 
#     sess.run(maintain_averages_op) 
#     print (sess.run([v1, ema.average(v1)])) # 更新step和v1的取值，衰减率为0.99，所以v1的滑动平均被更新为0.99*4.5+0.01*10=4.555 
#     # sess.run(tf.assign(step, 10)) 
#     # sess.run(tf.assign(v1, 10)) 
#     a = maintain_averages_op 
#     print (sess.run([v1, ema.average(v1)]))
#     # 更新一次v1的滑动平均值 
#     # sess.run(maintain_averages_op) 
#     print (sess.run([v1, ema.average(v1)]))
#     # print(sess.run(v1))
#     # sess.run(maintain_averages_op) 
#     # print(sess.run(v1))
#     # print (sess.run([v1, ema.average(v1)]))
#     # sess.run(tf.assign(v1, 100)) 
    