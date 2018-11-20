import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import os

# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 运行次数
max_step = 1001
# 图片的数量
image_num = 3000
# 文件路径
DIR = str(os.getcwd())

# 定义会话
sess = tf.Session()

# 载入图片，把一定数量的测试图片载入内存
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')


# 参数摘要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        # 平均值
        tf.summary.scalar('mean', mean)
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 标准差
        tf.summary.scalar("stddev", stddev)
        # 最大值
        tf.summary.scalar('max', tf.reduce_max(var))
        # 最小值
        tf.summary.scalar("min", tf.reduce_min(var))
        # 直方图
        tf.summary.histogram('histogram', var)


# 命名空间
with tf.name_scope('input'):
    # 这里None表示是第一个维度可以是任意长度
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    # 正确的标签
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

# 显示图片
with tf.name_scope("input_reshape"):
    # -1表示任意值,1表示黑白，将784转化为28x28
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    # 这里10表示的是10张图片
    tf.summary.image('input', image_shaped_input, 10)

with tf.name_scope('layer'):
    # 创建一个神经网络
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    # 交叉熵代价函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 初始化变量
sess.run(tf.global_variables_initializer())

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果放入一个布尔型的列表中
        # argmax返回一维张量中最大值所在的位置
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        # 把correct_prediction变为float32
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 生成metadata文件
if tf.gfile.Exists(DIR + r'\projector\projector\metadata.tsv'):
    pass
    #tf.gfile.DeleteRecursively(DIR + r'\projector\projector\metadata.tsv')
with open(DIR + r'\projector\projector\metadata.tsv', 'w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
    for i in range(image_num):
        f.write(str(labels[i]) + '\n')

# 合并所有的summary
merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(DIR + r'\projector\projector', sess.graph)
saver = tf.train.Saver()
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = embedding.name
embed.metadata_path = DIR + r'\projector\projector\metadata.tsv'
embed.sprite.image_path = DIR + r'\projector\data\mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28, 28])
projector.visualize_embeddings(projector_writer, config)

for i in range(max_step):
    # 每个批次为100样本
    batch_xs, batch_ys = mnist.train.next_batch(100)
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary, i)

    if i % 100 == 0:
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('Iter ' + str(i) + "Testing Accuracy= " + str(acc))
saver.save(sess, DIR + r'\projector\projector\a_model.ckpt', global_step=max_step)
projector_writer.close()
sess.close()
