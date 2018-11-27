import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_train
import mnist_interface

eval_sec = 3
# global_step = tf.Variable(0,trainable = False )
def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32 ,[None, mnist_interface.input_node] , name = 'x-input')
        _y = tf.placeholder(tf.float32 ,[None, mnist_interface.output_node] , name = 'y-input')
        v_feed = {x:mnist.validation.images,_y:mnist.validation.labels}
        y = mnist_interface.interface(x,None)
        correct_pridict = tf.equal(tf.argmax(y,1),tf.argmax(_y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pridict,tf.float32))
        # variable_average = tf.train.ExponentialMovingAverage(mnist_train.moving_average_decay)
        # variable_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver()
        with tf.Session() as sess:
                # print('hey')
                ckpt = tf.train.get_checkpoint_state(mnist_train.model_path)
                c = ckpt.all_model_checkpoint_paths
                for i in c:
                        try:
                                # print('hey')
                                saver.restore(sess,i)
                                # print(ckpt and ckpt.model_checkpoint_path)
                                time.sleep(eval_sec)
                                global_step = i.split('-')[-1]
                                accuracy_score = sess.run(accuracy,feed_dict = v_feed)
                                print('after {} correct {}'.format(global_step,accuracy_score))
                        except:
                                print('no check point found')
                                time.sleep(eval_sec)
def main(argv=None):
    mnist = input_data.read_data_sets('C:\\Users\\user\\Documents\\GitHub\\tensorflow\\generation-4\\mnist_data',one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
        tf.app.run()