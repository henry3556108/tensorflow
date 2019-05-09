from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("C:\\Users\\user\\Documents\\GitHub\\tensorflow",one_hot=True)
x = tf.placeholder(tf.float32 ,[None, 784] , name = 'x-input')
_y = tf.placeholder(tf.float32 ,[None, 10] , name = 'y-input')
sess = tf.Session()
saver = tf.train.Saver({x:x,_y:_y})
test_feed = {x : mnist.test.images , _y : mnist.test.labels}
mnist = input_data.read_data_sets("C:\\Users\\user\\Documents\\GitHub\\tensorflow\\generation-3",one_hot=True)
test_feed = {x : mnist.test.images , _y : mnist.test.labels}
saver.restore(sess, 'C:\\Users\\user\\Documents\\GitHub\\tensorflow\\generation-3\\test-model.ckpt')
print('model is restore')
sess.run(feed_dict = test_feed)