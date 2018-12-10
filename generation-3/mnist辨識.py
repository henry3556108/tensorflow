from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
input_node = 784 #這個是每張mnist數字的像素 28*28
output_node = 10 #我們要將圖片分類成0~9 總共十項
layer1_node= 500 #這個是隱藏層的節點 總共有500個
batch_size = 100 #每次訓練有100筆資料進去
learning_rate_base = 0.8 #一開始的學習率
learning_rate_decay = 0.99 #學習率的衰減 一開始接近1 越到後面會越小 控制學習率 避免overfitting
regularization_rate = 0.0001 # 這是正規化的lambda
training_steps = 3000 # 訓練量
moving_average_decay = 0.99 #滑動平均衰減率

def interface(input_tensor,avg_class,weight1,basies1,weight2,basies2,):
    if avg_class == None:#沒有滑動平均模型
        layer1 = tf.nn.leaky_relu(tf.matmul(input_tensor,weight1) + basies1)
        return tf.nn.leaky_relu(tf.matmul(layer1,weight2) + basies2)
    else:#有滑動平均模型
        layer1 = tf.nn.leaky_relu(tf.matmul(input_tensor,avg_class.average(weight1)) + avg_class.average(basies1))
        return tf.nn.leaky_relu(tf.matmul(layer1,avg_class.average(weight2)) + avg_class.average(basies2))

def train(mnist):
    x = tf.placeholder(tf.float32 ,[None, input_node] , name = 'x-input')
    _y = tf.placeholder(tf.float32 ,[None, output_node] , name = 'y-input')
    #製造權重與閥值
    weight1 = tf.Variable(tf.truncated_normal([input_node,layer1_node],stddev = 0.1))
    basies1 = tf.Variable(tf.constant(0.1,shape = [layer1_node]))
    weight2 = tf.Variable(tf.truncated_normal([layer1_node,output_node],stddev = 0.1))
    basies2 = tf.Variable(tf.constant(0.1,shape=[output_node]))
    y=interface(x,None,weight1,basies1,weight2,basies2)
    global_step = tf.Variable(0,trainable = False )# 控制平滑衰減率 decay= min（decay，（1+steps）/（10+steps））
    variable_average = tf.train.ExponentialMovingAverage(moving_average_decay,global_step) #滑動平均模型的函式
    variable_average_op = variable_average.apply(tf.trainable_variables()) #將所有trainable的值都丟進去這個滑動平均裡面=>看interface的else
    average_y = interface(x, variable_average , weight1, basies1, weight2, basies2)
    #書上說 如果問題只有一個正解,那可以使用sparse_softmax_cross_entropy_with_logits 可以加快交叉熵的計算
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels=tf.argmax(_y,1)) #tf.argmax(vector, 1)：返回的是vector中的最大值的索引 在_y裡面只有0跟1 所以他會回傳1的索引 y則是我們經過運算的值 要拿來跟_y計算損失函數
    cross_entropy_mean = tf.reduce_mean(cross_entropy) #我們拿來取平均值
    regularize = tf.contrib.layers.l2_regularizer(regularization_rate)# 使用L2正規化來計算
    loss = regularize(weight1) + regularize(weight2) + cross_entropy_mean # 我們的損失函數加起來 每一個都會是一個值 不會是一個矩陣
    learning_rate = tf.train.exponential_decay(learning_rate_base,global_step,mnist.train.num_examples/batch_size,learning_rate_decay) #　這個是控制學習率的衰減　需要給他初值　訓練的起點終點　學習率的衰減率
    train_step = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step) #在這裡定義我們的模型是甚麼
    train_op = tf.group(train_step,variable_average_op) # 讓他一次完成多個操作 可以反向傳播優化 又可以取滑動平均值  
    # 這邊在計算它的正確率
    correct_pridict = tf.equal(tf.argmax(average_y,1),tf.argmax(_y,1))
    accuracy =tf.reduce_mean(tf.cast(correct_pridict,tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run() #初始化所有的varialbe
        validate_feed = {x : mnist.validation.images, _y : mnist.validation.labels}# 這個是驗證資料
        test_feed = {x : mnist.test.images , _y : mnist.test.labels}
        for i in range(training_steps):
            if i %1000 ==0:
                correct = sess.run(accuracy,feed_dict = validate_feed)
                print('訓練了{}次之後 正確率為{}'.format(i , correct))
            xs,ys = mnist.train.next_batch(batch_size)
            sess.run(train_op,feed_dict = {x : xs,_y : ys})
        test_correct = sess.run(accuracy, feed_dict = test_feed)
        print('經過了{}次之後 正確率來到了{}'.format(training_steps,test_correct))
        # print('weight1是: ',sess.run(weight1),'\nweight2是: ',sess.run(weight2))
        saver = tf.train.Saver()
        # saver.export_meta_graph("C:\\Users\\user\\Documents\\GitHub\\tensorflow\\generation-3\\model.ckpt",as_text = True)
        saver.save(sess, 'C:\\Users\\user\\Documents\\GitHub\\tensorflow\\generation-3\\test-model.ckpt')

def main():
    mnist = input_data.read_data_sets("C:\\Users\\user\\Documents\\GitHub\\tensorflow",one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()