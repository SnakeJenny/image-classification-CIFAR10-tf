'''Builds a 2-layer fully-connected neural network'''
# inference（）构成通过网络的正向传递并返回类分数。
# loss（）比较预测和真实的类分数并生成损失值。
# training（）执行训练步骤，并优化模型的内部参数。
# evaluation（）测量模型的性能。

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def inference(images, image_pixels, hidden_units, classes, reg_constant=0):
  '''Build the model up to where it may be used for inference.
  inference（）函数接收输入图像并返回类分数,这是一个训练有素的分类器需要做的。
  但为了得到一个训练有素的分类器，首先需要测量这些类分数表现有多好，这是损失函数要做的工作。
  Args:
      images: Images placeholder (input data).
      image_pixels: Number of pixels per image.
      hidden_units: Size of the first (hidden) layer.
      classes: Number of possible image classes/labels.
      reg_constant: Regularization constant (default 0).

  Returns:
      logits: Output tensor containing the computed logits.
  '''

  # Layer 1
  with tf.variable_scope('Layer1'):
    # Define the variables
    # 关于单层中的神经元，它们都接收完全相同的输入值，如果它们都具有相同的内部参数，
    # 则它们将进行相同的计算并且输出相同的值。为了避免这种情况，需要随机化它们的初始权重。
    # 我们使用了一个通常可以很好运行的初始化方案，将weights初始化为正态分布值。
    # 丢弃与平均值相差超过2个标准偏差的值，并且将标准偏差设置为输入像素数量的平方根的倒数。
    weights = tf.get_variable(
      name='weights',
      shape=[image_pixels, hidden_units],
      initializer=tf.truncated_normal_initializer(
        stddev=1.0 / np.sqrt(float(image_pixels))),
      regularizer=tf.contrib.layers.l2_regularizer(reg_constant)
    )

    biases = tf.Variable(tf.zeros([hidden_units]), name='biases')

    # Define the layer's output
    hidden = tf.nn.relu(tf.matmul(images, weights) + biases)

  # Layer 2
  # 第2层与第1层非常相似，其输入值为hidden_units，输出值为classes，因此weights矩阵的
  # 维度是[hidden_units，classes]。 由于这是我们网络的最后一层，所以不再需要ReLU。
  with tf.variable_scope('Layer2'):
    # Define variables
    weights = tf.get_variable('weights', [hidden_units, classes],
      initializer=tf.truncated_normal_initializer(
        stddev=1.0 / np.sqrt(float(hidden_units))),
      regularizer=tf.contrib.layers.l2_regularizer(reg_constant))

    biases = tf.Variable(tf.zeros([classes]), name='biases')

    # Define the layer's output
    # 通过将输入（hidden）互乘以weights，再加上bias就可得到类分数（logits）
    logits = tf.matmul(hidden, weights) + biases

    # 通过定义一个汇总操作告诉TensorFlow收集某些张量,
    # 本例中logits，loss和accuracy的摘要信息。汇总操作的其他参数就只是一些想要添加到总结的标签。

    # Define summery-operation for 'logits'-variable
    # tf.histogram_summary（）允许我们记录logits变量的值，收集有关的多个值分布信息,
    # 以便以后用TensorBoard进行分析。
    tf.summary.histogram('logits', logits)

  return logits


def loss(logits, labels):
  '''Calculates the loss from logits and labels.
  我们计算logits（模型的输出）和labels（来自训练数据集的正确标签）之间的交叉熵，
  这已经是我们对softmax分类器的全部损失函数，但是这次我们想要使用正则化，
  所以必须给损失添加另一个项。
  Args:
    logits: Logits tensor, float - [batch size, number of classes].
    labels: Labels tensor, int64 - [batch size].

  Returns:
    loss: Loss tensor of type float.
  '''

  with tf.name_scope('Loss'):
    # Operation to determine the cross entropy between logits and labels
    cross_entropy = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy'))

    # Operation for the loss function
    loss = cross_entropy + tf.add_n(tf.get_collection(
      tf.GraphKeys.REGULARIZATION_LOSSES))

    # Add a scalar summary for the loss
    # 使用scalar_summary记录有关标量（非矢量）值
    tf.summary.scalar('loss', loss)

  return loss


def training(loss, learning_rate):
  '''Sets up the training operation.

  Creates an optimizer and applies the gradients to all trainable variables.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_step: The op for training.
  '''

  # Create a variable to track the global step
  # 优化程序会在每次迭代时自动递增global_step参数，为了实时显示训练过程。
  global_step = tf.Variable(0, name='global_step', trainable=False)

  # Create a gradient descent optimizer
  # (which also increments the global step counter)
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    loss, global_step=global_step)

  return train_step


def evaluation(logits, labels):
  '''Evaluates the quality of the logits at predicting the label.
  模型精度的计算与softmax情况相同：将模型的预测与真实标签进行比较，并计算正确预测的频率。
  Args:
    logits: Logits tensor, float - [batch size, number of classes].
    labels: Labels tensor, int64 - [batch size].

  Returns:
    accuracy: the percentage of images where the class was correctly predicted.
  '''

  with tf.name_scope('Accuracy'):
    # Operation comparing prediction with true label
    correct_prediction = tf.equal(tf.argmax(logits,1), labels)

    # Operation calculating the accuracy of the predictions
    accuracy =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Summary operation for the accuracy
    # 我们还对随着时间的推移精度如何演变感兴趣，因此添加了一个跟踪accuracy的汇总操作。
    tf.summary.scalar('train_accuracy', accuracy)

  return accuracy
