import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 128
n_batch = mnist.train.num_examples // batch_size
max_step = 1000
keep_ = 0.8
log_dir = "Logs"


# 生成权重
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')


# 生成偏差
def bias_vairable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), name='b')


# 记录变量
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='conv2d')


def conv_layer(input_tensor, weight_shape, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable(weight_shape)
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_vairable([weight_shape[-1]])
            variable_summaries(biases)
        with tf.name_scope('conv_comput'):
            preactivate = conv2d(input_tensor, weights) + biases
        with tf.name_scope('activate'):
            activations = act(preactivate)
        return activations


def linear_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_vairable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('linear_comput'):
            preactivate = tf.matmul(input_tensor, weights) + biases
        with tf.name_scope('activate'):
            activations = act(preactivate)
        return activations


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='Max_pool')


with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, [None, 784], name='input_x')
    with tf.name_scope('Input_reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x-image')
        tf.summary.image('input', x_image, 10)
    y = tf.placeholder(tf.float32, [None, 10], name='input_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# 第一次卷积   28*28*1->28*28*32
conv_layer1 = conv_layer(x_image, [5, 5, 1, 32], 'conv_layer1')
# 池化之后变为 14*14*32
with tf.name_scope('Max_pool1'):
    h_pool1 = max_pool_2x2(conv_layer1)

# 第二次卷积 14*14*32->14*14*64
conv_layer2 = conv_layer(h_pool1, [5, 5, 32, 64], 'conv_layer2')
# 第二次池化之后变为 7*7*64
with tf.name_scope('Max_pool2'):
    h_pool2 = max_pool_2x2(conv_layer2)

with tf.name_scope('Flatten'):
    flatten_ = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# 第一个全连接层 7*7*64 - 1024
fc1 = linear_layer(flatten_, 7 * 7 * 64, 1024, 'FC1')


with tf.name_scope('Dropput'):
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

# 第二个全连接层 1024 - 10
logits = linear_layer(fc1_drop, 1024, 10, 'FC2', act=tf.nn.sigmoid)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.name_scope('accuracy'):
    prediction = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()


def get_dict(train):
    if train:
        xs, ys = mnist.train.next_batch(batch_size)
        k = keep_
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y: ys, keep_prob: k}


with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    sess.run(tf.global_variables_initializer())

    for i in range(max_step):
        if i % 10 == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict=get_dict(False))
            test_writer.add_summary(summary, i)
            print("Step: " + str(i) + ", acc: " + str(acc))
        else:
            summary, _ = sess.run([merged, train_step], feed_dict=get_dict(True))
            train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()
