import tensorflow as tf
import hashlib as hs
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from numpy import *
import numpy as np 
import mnist_train

print(tf.__version__)

# input = tf.Variable(tf.random_normal([1,28,28,1]))
# filter = tf.Variable(tf.random_normal([5,5,1,32]))
 
# op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

# with tf.Session() as sess:
#     tf.initialize_all_variables().run()
#     print(np.shape(sess.run(op)))
#     ckpt = tf.train.get_checkpoint_state(mnist_train.model_path)
#     print(ckpt)
# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
# output = tf.multiply(input1, input2)
# saver = tf.train.Saver()
# ckpt = tf.train.get_checkpoint_state(mnist_train.model_path)
# c=ckpt.model_checkpoint_path
# variable_average = tf.train.ExponentialMovingAverage(mnist_train.moving_average_decay)
# variable_to_restore = variable_average.variables_to_restore()
# saver = tf.train.Saver(variable_to_restore)
# with tf.Session() as sess:
#   saver.restore(sess,"C:\\Users\\user\\Documents\\GitHub\\tensorflow\\generation-4\\check\\mnist_ckpt-18001")
        
# print(ckpt and "C:\\Users\\user\\Documents\\GitHub\\tensorflow\\generation-4\\check\\mnist_ckpt-18001")
# # print(ckpt and c)
# # print(c)
# # global_step = ckpt.model_checkpoint_path.split('-')[-1]
# # print(global_step)