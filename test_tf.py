# 11、手把手教你用TensorFlow搭建图像识别系统
# 根据TensorFlow代码规范，在所有TensorFlow Python文件中为了Python2和3的兼容性，都应该添加future语句。
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 然后导入TensorFlow，numpy用于数值计算和时间模块。data_helper.py包括加载和准备数据集的函数。
import numpy as np
import tensorflow as tf
import time
import data_helpers

# 我们启动一个计时器测量运行时间和定义一些参数。
beginTime = time.time()

# Parameter definitions
batch_size = 100
learning_rate = 0.005
max_steps = 1000

# Prepare data
# 因为读取数据并不是我们要做的核心，我把这部分的函数单独放在data_helper.py文件中。它只是负责读取包含数据集的文件，并把数据放入一个方便我们操作的数据结构中。
# load_data()返回一个dictionary类型数据：
# images_train：训练集转换为50000x3072（32像素x32像素x3个颜色通道）的数组
# labels_train:训练集的50000个标签（每个数字从0到9代表图像训练集的10个分类）
# images_test：测试集（10000x3072）
# labels_test：测试集的10000个标签
# classes：10个文本标签，将数字转换成文字（0代表“飞机”，1代表“车”，等等）
data_sets = data_helpers.load_data()

# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])


# Define variables (these are the values we want to optimize)
weights = tf.Variable(tf.zeros([3072, 10]))
biases = tf.Variable(tf.zeros([10]))

# Define the classifier's result
# 得到计算的分数，softmax将该分数转化为0～1的概率值
logits = tf.matmul(images_placeholder, weights) + biases

# Define the loss function
# 我们比较模型预测值logits和正确分类值labels_placeholder。sparse_softmax_cross_entropy_with_logits()函数的输出就是每幅输入图像的损失值。然后我们只需计算输入图像的平均损失值。
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
  labels=labels_placeholder))
# Define the training operation
# 也可以尝试使用ADAM这种最优化算法
# 自动分化（auto-differentiation）的技术，它可以计算出相对于参数值，损失值i梯度。这就是说它可以知道每个参数对总的损失的影响，小幅度的加或减参数是否可以降低损失。然后依此调整所有参数值，增加模型的准确性。
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Operation comparing prediction with true label
# logits的argmax返回分数最高的分类。这就是预测的分类标签。tf.equal()将这个标签与正确的分类标签相比较，然后返回布尔向量。
correct_prediction = tf.equal(tf.argmax(logits, 1), labels_placeholder)

# Operation calculating the accuracy of our predictions
# 布尔数转换为浮点数（每个值不是0就是1），这些数求平均得到的分数就是正确预测图像的比例。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# -----------------------------------------------------------------------------
# Run the TensorFlow graph
# -----------------------------------------------------------------------------

with tf.Session() as sess:
    # Initialize variables
    # 运行这个会话控制的第一步就是初始化我们早先创建的变量。
    sess.run(tf.initialize_all_variables())

    # Repeat max_steps times
    # 开始迭代训练过程。它会重复进行max_steps次。
    for i in range(max_steps):

        # Generate input data batch
        # 一次epoch（全数据集）尽可能增大batch，加速算法的收敛。1. batch过大，内存受限，下降方向可能不会发生变化，单次迭代过慢导致参数更新变缓；2. batch过小，在震荡区游荡无法收敛。
        # batch_size在从0到整个训练集的大小之间随机指定一个值。然后根据这个值，批处理选取相应个数的图像和标签。
        indices = np.random.choice(data_sets['images_train'].shape[0], batch_size)
        images_batch = data_sets['images_train'][indices]
        labels_batch = data_sets['labels_train'][indices]

        # Periodically print out the model's current accuracy
        # 每100次迭代，我们对模型训练数据批的当前精确率进行检查。我们只需要调用我们之前定义的精确率操作来完成。
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                      images_placeholder: images_batch, labels_placeholder: labels_batch})
            print('Step {:5d}: training accuracy {:g}'.format(i, train_accuracy))

        # Perform a single training step
        # 整个训练循环中最重要的一行代码。我们告诉模型执行一个单独的训练步骤。我们没有必要为了参数更新再次声明模型需要做什么。所有的信息都是由TensorFlow图表中的定义提供的。TensorFlow知道根据损失使用梯度下降算法更新参数。而损失依赖logits。Logits又依靠weights，biases和具体的输入批。因此我们只需要向模型输入训练数据批。这些通过提供查找表来完成。训练数据批已经在我们早先定义的占位符中完成了赋值。
        sess.run(train_step, feed_dict={images_placeholder: images_batch,labels_placeholder: labels_batch})

    # After finishing the training, evaluate on the test set
    # 训练结束后，我们用测试集对模型进行评估。这是模型第一次见到测试集。所以测试集中的图像对模型来时是全新的。我们会评估训练后的模型在处理从未见过的数据时表现如何。
    test_accuracy = sess.run(accuracy, feed_dict={
          images_placeholder: data_sets['images_test'],
            labels_placeholder: data_sets['labels_test']})
    print('Test accuracy {:g}'.format(test_accuracy))

# 打印出训练和运行模型用了多长时间
endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))
