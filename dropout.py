import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 定义每个批次的大小
batch_size = 100

# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob=tf.placeholder(tf.float32)
lr = tf.Variable(0.001, dtype=tf.float32)

# # 创建一个简单的神经网络
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 创建一个有一层隐含层的神经网络
W_1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b_1 = tf.Variable(tf.zeros([500])+0.1)
L_1 = tf.nn.tanh(tf.matmul(x, W_1) + b_1)
L_1_drop = tf.nn.dropout(L_1, keep_prob)


W_2 = tf.Variable(tf.truncated_normal([500,300], stddev=0.1))
b_2 = tf.Variable(tf.zeros([300])+0.1)
L_2 = tf.nn.tanh(tf.matmul(L_1, W_2) + b_2)
L_2_drop = tf.nn.dropout(L_2, keep_prob)

W_3 = tf.Variable(tf.truncated_normal([300,10], stddev=0.1))
b_3 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L_2, W_3) + b_3)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))

# 改用对数似然代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用梯度下降法
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果的准确性存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(50):
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})

        print("Iter " + str(epoch) + ", Testing Accuracy " + str(test_acc), "Training Accuracy " + str(train_acc), "Learning Rate " + str(lr))
