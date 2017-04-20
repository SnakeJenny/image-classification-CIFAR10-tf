'''Trains and evaluates a fully-connected neural net classifier for CIFAR-10'''
# 这里介绍run_fc_model.py如何运行、训练和评估模型

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os.path
import data_helpers
import two_layer_fc

# Model parameters as external flags
# 在强制导入之后，将模型参数定义为外部标志。
# TensorFlow有自己的命令行参数模块，这是一个围绕Python argparse的小封装包。
# 在这里使用它是为了方便，但也可以直接使用argparse。

# 定义了命令行参数。每个标志的参数是标志的名称（其默认值和一个简短的描述）。
# 使用-h标志执行文件将显示这些描述。
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for the training.')
flags.DEFINE_integer('max_steps', 20000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 120, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('batch_size', 400,
  'Batch size. Must divide dataset sizes without remainder.')
flags.DEFINE_string('train_dir', '/usr/eli/dev/logdir',
  'Directory to put the training data.')
flags.DEFINE_float('reg_constant', 0.1, 'Regularization constant.')

# 调用实际解析命令行参数的函数，然后将所有参数的值打印到屏幕上。
FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
  print('{} = {}'.format(attr, value))
print()

IMAGE_PIXELS = 3072
CLASSES = 10

beginTime = time.time()

# Put logs for each run in separate directory
# TensorBoard要求每次运行的日志都位于单独的目录中，
# 因此将日期和时间信息添加到日志目录的名称地址。
logdir = FLAGS.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'

# Uncommenting these lines removes randomness
# You'll get exactly the same result on each run
# np.random.seed(1)
# tf.set_random_seed(1)

# Load CIFAR-10 data
# 加载CIFAR-10数据，并返回包含独立训练和测试数据集的字典。
data_sets = data_helpers.load_data()

# -----------------------------------------------------------------------------
# Prepare the Tensorflow graph
# (We're only defining the graph here, no actual calculations taking place)
# -----------------------------------------------------------------------------

# Define input placeholders
# images_placeholder将每张图片批处理成一定尺寸乘以像素的大小，
# 批处理大小设定为“None”允许运行图片时可随时设定大小。
# 用于训练网络的批处理大小可以通过命令行参数设置，但是对于测试，我们将整个测试集作为一个批处理
images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS],
  name='images')
labels_placeholder = tf.placeholder(tf.int64, shape=[None], name='image-labels')

# Operation for the classifier's result
logits = two_layer_fc.inference(images_placeholder, IMAGE_PIXELS,
  FLAGS.hidden1, CLASSES, reg_constant=FLAGS.reg_constant)

# Operation for the loss function
loss = two_layer_fc.loss(logits, labels_placeholder)

# Operation for the training step
train_step = two_layer_fc.training(loss, FLAGS.learning_rate)

# Operation calculating the accuracy of our predictions
accuracy = two_layer_fc.evaluation(logits, labels_placeholder)

# Operation merging summary data for TensorBoard
# 为TensorBoard定义一个summary操作函数
summary = tf.summary.merge_all()

# Define saver to save model state at checkpoints
# 生成一个保存对象以保存模型在检查点的状态，
# 处理不同会话（session）中任何与文件系统有持续数据传输的交互，
# 构造函数含有三个部分：目标target，图graph和配置config。
# 保存/恢复之后，它需要知道的唯一的事情是使用哪个图和变量。
saver = tf.train.Saver()

# -----------------------------------------------------------------------------
# Run the TensorFlow graph
# -----------------------------------------------------------------------------

with tf.Session() as sess:
  # Initialize variables and create summary-writer
  sess.run(tf.global_variables_initializer())
  # 创建一个汇总编辑器，使其定期将日志信息保存到磁盘。
  summary_writer = tf.summary.FileWriter(logdir, sess.graph)

  # Generate input data batches
  # 负责生成批输入数据。让我们假设我们有100个训练图像，批次大小为10。
  # 在softmax示例中，我们只为每次迭代选择了10个随机图像，特别注意是随机。
  zipped_data = zip(data_sets['images_train'], data_sets['labels_train'])

  # 对训练数据集的100个图像随机混洗。混洗之后的数据的前10个图像作为我们的第一个批次，
  # 接下来的10个图像是我们的第二批，后面的批次以此类推。10批后，在数据集的末尾，再重复混洗过程
  batches = data_helpers.gen_batch(list(zipped_data), FLAGS.batch_size,
    FLAGS.max_steps)

  for i in range(FLAGS.max_steps):

    # Get next input data batch
    batch = next(batches)
    images_batch, labels_batch = zip(*batch)
    feed_dict = {
      images_placeholder: images_batch,
      labels_placeholder: labels_batch
    }

    # Periodically print out the model's current accuracy
    if i % 100 == 0:
      train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
      print('Step {:d}, training accuracy {:g}'.format(i, train_accuracy))
      summary_str = sess.run(summary, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, i)

    # Perform a single training step
    sess.run([train_step, loss], feed_dict=feed_dict)

    # Periodically save checkpoint
    if (i + 1) % 1000 == 0:
      checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
      saver.save(sess, checkpoint_file, global_step=i)
      print('Saved checkpoint')

  # After finishing the training, evaluate on the test set
  test_accuracy = sess.run(accuracy, feed_dict={
    images_placeholder: data_sets['images_test'],
    labels_placeholder: data_sets['labels_test']})
  print('Test accuracy {:g}'.format(test_accuracy))

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))
