import tensorflow as tf

x = tf.Variable([1, 2])
a = tf.constant([3, 3])

# 增加一个减法op
sub = tf.subtract(x, a)

# 增加一个加法op
add = tf.add(x, sub)

# 变量初始化op
init_op = tf.global_variables_initializer()


#
# with tf.Session() as sess:
#     sess.run(init_op)
#     print(sess.run(sub))
#     print(sess.run(add))


# 创建一个变量初始化为0
state = tf.Variable(0, name='counter')
# 创建一个op，作用是让state加1
new_value = tf.add(state, 1)
# 赋值op
update = tf.assign(state, new_value)
# 变量初始化
init = tf.global_variables_initializer()

with tf.Session() as sese:
    sese.run(init)
    print(sese.run(state))
    for _ in range(5):
        sese.run(update)
        print(sese.run(state))
