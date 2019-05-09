import os
import tensorflow as tf
import mnist_interface as m_inter
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 100 #每次訓練有100筆資料進去
learning_rate_base = 0.8 #一開始的學習率
learning_rate_decay = 0.99 #學習率的衰減 一開始接近1 越到後面會越小 控制學習率 避免overfitting
regularization_rate = 0.0001 # 這是正規化的lambda
training_steps = 5000 # 訓練量
moving_average_decay = 0.99 #滑動平均衰減率
model_path = 'C:\\Users\\user\\Documents\\GitHub\\tensorflow\\generation-4\\check'
model_name = 'mnist_ckpt'

def train(mnist):
    x = tf.placeholder(tf.float32 ,[None, m_inter.input_node] , name = 'x-input')
    _y = tf.placeholder(tf.float32 ,[None, m_inter.output_node] , name = 'y-input')
    regularize = tf.contrib.layers.l2_regularizer(regularization_rate)# 使用L2正規化來計算
    global_step = tf.Variable(0,trainable = False)# 控制平滑衰減率 decay= min（decay，（1+steps）/（10+steps）） 也記錄現在進入底幾次訓練
    variable_average = tf.train.ExponentialMovingAverage(moving_average_decay,global_step) #滑動平均模型的函式
    variable_average_op = variable_average.apply(tf.trainable_variables()) #將所有trainable的值都丟進去這個滑動平均裡面=>看interface的else
    y = m_inter.interface(x,regularize)
    #書上說 如果問題只有一個正解,那可以使用sparse_softmax_cross_entropy_with_logits 可以加快交叉熵的計算
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels=tf.argmax(_y,1)) #tf.argmax(vector, 1)：返回的是vector中的最大值的索引 在_y裡面只有0跟1 所以他會回傳1的索引 y則是我們經過運算的值 要拿來跟_y計算損失函數
    cross_entropy_mean = tf.reduce_mean(cross_entropy) #我們拿來取平均值
    loss =tf.add_n(tf.get_collection('losses')) + cross_entropy_mean # 將剛剛加入進去'losses'的正規畫損失與我們的損失函數加起來 每一個都會是一個值 不會是一個矩陣
    #tf.add_n([p1, p2, p3....])函数是实现一个列表的元素的相加。就是输入的对象是一个列表，列表里的元素可以是向量，矩阵，
    learning_rate = tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples/batch_size,learning_rate_decay) #　這個是控制學習率的衰減　需要給他初值　訓練的起點終點　學習率的衰減率
    train_step = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step) #在這裡定義我們的模型是甚麼
    #with tf.control_dependencies([train_op,variables_average_op])
    #train_op = tf.no_op(name'train')這個與下面那行有一樣的功用 可是我看不懂他在寫啥
    train_op = tf.group(train_step,variable_average_op) # 讓他一次完成多個操作 可以反向傳播優化 又可以取滑動平均值  
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(training_steps):
            xs,ys = mnist.train.next_batch(batch_size)
            _ ,loss_value ,step = sess.run([train_op,loss,global_step],feed_dict = {x : xs ,_y : ys})
            if i % 2500 == 0:
                print('after {} train, now loss_value is {}'.format(step,loss_value))
                saver.save(sess,os.path.join(model_path,model_name),global_step = global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("C:\\Users\\user\\Documents\\GitHub\\tensorflow\\generation-4\\mnist_data",one_hot = True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()