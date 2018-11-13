import tensorflow as tf
var=tf.Variable(20)
init = tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
print(sess.run(var))