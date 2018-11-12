import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from numpy.random import RandomState
import numpy as np
import tensorflow as tf


batch_size=8
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x = tf.placeholder(tf.float32,shape = (None,2),name = 'x_input')
y_ = tf.placeholder(tf.float32,shape = (None),name = 'y_input')

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

cross_entropy = -(tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))))
trainstep = tf.train.AdadeltaOptimizer(3.5).minimize(cross_entropy)
rdm = RandomState(1)
dataset_size=128

X = rdm.rand(dataset_size , 2)

Y = [[int(x1+x2 < 1) for (x1 , x2) in X]]
dic={}
with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    print(sess.run(w1),'\n',sess.run(w2))
    STEP = 5000
    for i in range(STEP):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)
        sess.run(trainstep,feed_dict={x:X[start:end] , y_:Y[start:end]})
        
        if i %1000 == 0:
            cross=sess.run(cross_entropy,feed_dict={x:X,y_:Y})
            print(i,'\n',cross)
            
    print(sess.run(w1),'\n',sess.run(w2))
