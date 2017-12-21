## TF实现神经网络举例

本节我们主要看一下TensorFlow如何愉快的做神经网络！神经网络的结构及训练过程我就不说了，我们现在可以用Python中的Numpy独立的完成神经网路的设计过程，那么如何用tensorflow完成呢？

### 1.TF训练DNN


老规矩直接解释代码：

法1：以一种简单的形式


```python
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
import numpy as np
sess = tf.Session()

def fully_connected(input_layer, weights, biases):
	#1st
	layer = tf.add(tf.matmul(input_layer, weights), biases)
	return(tf.nn.relu(layer))

	# 2 (25 hidden nodes)
	weight_1 = init_weight(shape=[8, 25], st_dev=10.0)
	bias_1 = init_bias(shape=[25], st_dev=10.0)
	layer_1 = fully_connected(x_data, weight_1, bias_1)

	# 3 (10 hidden nodes)
	weight_2 = init_weight(shape=[25, 10], st_dev=10.0)
	bias_2 = init_bias(shape=[10], st_dev=10.0)
	layer_2 = fully_connected(layer_1, weight_2, bias_2)
	
	# 4 (3 hidden nodes)
	weight_3 = init_weight(shape=[10, 3], st_dev=10.0)
	bias_3 = init_bias(shape=[3], st_dev=10.0)
	layer_3 = fully_connected(layer_2, weight_3, bias_3)
	
	# output layer (1 output value)
	weight_4 = init_weight(shape=[3, 1], st_dev=10.0)
	bias_4 = init_bias(shape=[1], st_dev=10.0)
	final_output = fully_connected(layer_3, weight_4, bias_4)

loss = tf.reduce_mean(tf.abs(y_target - final_output))
my_opt = tf.train.AdamOptimizer(0.05)
train_step = my_opt.minimize(loss)
init = tf.initialize_all_variables()
sess.run(init)

# Initialize the loss vectors
loss_vec = []
test_loss = []
for i in range(200):
	# 划分训练集
	rand_index = np.random.choice(len(x_vals_train), size=batch_size)
	
	rand_x = x_vals_train[rand_index]
	rand_y = np.transpose([y_vals_train[rand_index]])
	# 训练模型
	sess.run(train_step, feed_dict={x_data: rand_x, y_target:rand_y})
	#loss
	temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target:rand_y})
	loss_vec.append(temp_loss)
	# test loss
	test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: np.transpose([y_vals_test])})
	test_loss.append(test_temp_loss)
	if (i+1)%25==0:
		print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))

# metrics的可视化
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss per Generation')

plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
```

上面的代码很简单，如果我们的DNN有50层，是不是我们要写好多变量，weight1,weight2,...，在def中要罗列好多变量，自己可能就晕掉了，在后期可视化模型时也不可以。并且我们并没有采取一些像正则化，变量的滑动平均（提高测试数据在模型中的稳健性，接触过时间序列可以对此有更深入的理解，这里不是我们的重点），dropout等的一些辅助优化技术，如果加上这些是不是这种办法显的更笨拙？

法2：对代代码做变量管理

```python
def inference(input_tensor,resuse=False):
	#1st
	with tf.variable_scope（"layer1",resuse=resuse):
		#根据传进来的resuse来判断是创建新变量还是使用已经创建好的，在第一次构造网络时
		#需要创建新的变量，以后每次调用这个函数直接使用resuse=True就不需要每次将变量传进来了
		weights = tf.get_variable('weights',[INPUT_NODE,LAYER1_NODE],
			initializer = tf.truncated_normal_initializer(stddev = 0.1))
		biases = tf.get_varible('biases',[LAYER1_NODE],initializer = tf.constant_initializer(0.))
		layer1 = tf.nn.relu(tf.matmul(input_tensor,weights) + biases)
	#类似的定义layer2
	with tf.variable_scope（"layer2",resuse=resuse):
		#根据传进来的resuse来判断是创建新变量还是使用已经创建好的，在第一次构造网络时
		#需要创建新的变量，以后每次调用这个函数直接使用resuse=True就不需要每次将变量传进来了
		weights = tf.get_variable('weights',[INPUT_NODE,OUTPUT_NODE],
			initializer = tf.truncated_normal_initializer(stddev = 0.1))
		biases = tf.get_varible('biases',[OUTPUT_NODE],initializer = tf.constant_initializer(0.))
		layer2 = tf.nn.relu(tf.matmul(input_tensor,weights) + biases)
	#返回最后的前向传播结果

	return layer2

x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
y = inference(x)

#在程序中使用训练好的神经网络进行推倒时，可以直接调用inference(new_x,True)
#当加入指数滑动平均等方法时，变量管理机制会会体现出更方便。
#注意：当resuse=True时，tf.get_variable函数将直接获取已经声明的变量，如果变量在命名空间中没有被声明过，此时会报错。

```


-----

### 2.TF模型的持久化

我们训练一个深层模型可能需要几周或几个月的时间，训练完了，如果不去保存下次使用还得几周的时间,模型的持久化就是解决这个问题，我们知道在MXNet和Keras中都有相应的机制，这也为迁移学习提供了方便。

TensorFlow提供了一个非常简单的API来保存和还原一个神经网络模型，这个API就是tf.train.Saver类

```python
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name='v2')

result = v1 + v2
init_op = tf.global_variables_initializer()

#声明tf.train.Saver类用于保存模型

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init_op)
	#将模型保存在C:/model/model.ckpt文件
	path = "C:/model/model.ckpt"
	saver.save(sess,path)
```

TF模型一般会保存在后缀为.ckpt(CheckPoint),虽然上面的程序指定了一个文件路径，但在这个文件路径下会出现三个文件，这是因为TensorFlow会将计算图的结构和图上参数取值分开来保存

+ model.ckpt.meta： 保存了计算图的结构（简单的可以理解为神经网络的网络结构）

+ model.ckpt: 这个文件保存了TensorFlow程序中每一个变量的取值

+ checkpoint文件：这个文件保存了一个目录下所有的模型文件列表

了解更多这些文件可搜索大量的介绍信息

+ 加载已经保存的TensorFlow模型

```python
import tensorflow as tf
#使用和保存模型代码中一样的方式来声明变量

v1 = tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name='v2')

result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
	#加载以保存的模型，并通过已保存的模型中的变量来计算加法
	saver.restore(sess,path)
	print sess.run(result)

```

区别：加载模型的代码中没有运行变量初始化过程，而是将变量的值通过已经保存的吗模型加载进来。

+ 如果不希望重复定义图上的运算，可以加载已经持久化的图

```python
import tensorflow as tf

#直接加载持久化的图
saver = tf.train.import_meta_graph(path+'.meta')
with tf.Session() as sess:
	saver.restore(ress,path)
	#通过张量的名称来获取张量
	print sess.run(tf.get_default_graph().get_tensor_by_name('add:0'))

#输出结果：[3.]
```

+ 保存或加载部分变量

默认加载和保存TF计算图上的所有变量，但有时可能只需要保存和加载部分变量，比如，我们有一个训练好的5层网络，但现在想尝试6层的网络，那么可以将前5层参数加载进来，训练最后一层网络即可（迁移学习）

tf.train.Saver类可以提供一个列表来指定需要保存和加载的变量，saver = tf.train.Saver([v1]),只有变量v1被加载进来

+ 变量重命名

tf.train.Saver支持保存和加载时的变量重命名

```python
v1 = tf.Variable(tf.constant(1.0,shape=[1]),name='other-v1')
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name='other-v2')

saver = tf.train.Saver({'v1':v1,'v2':v2})
#原来名称为v1的变量现在加载到变量v1中（名称为other-v1)。
```

+ 其他

tf.train.Saver会保存运行TF程序的全部信息，有时并不需要，比如在测试或离线预测时，我们只需知道神经网络的前向传播计算到输出就可以了，而不需要类似额变量初始化等辅助节点信息，在迁移学习中会遇到这些问题，于是TensorFlow提供了convert_variables_to _constants函数，通过这个函数可以将计算图中的变量及其取值通过常量的方式保存，这样TF计算图可以统一保存在一个文件中

```python
imort tensorflow as tf

from tensrflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
v2 = tf.Variable(tf.constant(2.0,shape=[2]),name='v2')

result = v1 + v2

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	#导出当前计算图的GraphDef部分，只需这与部分可以完成从输入到输出的计算过程
	graph_def  = tf.get_default_graph().as_graph_def()

	output_graph_def = graph_util.convert_variables_to_constants(sess,graph_def,['add'])
	#将导出的模型保存成文件
	with tf.gfile.GFile('C:/model.pb','wb') as f:
		f.write(output_graph_def.SerializeToString())
```

```python
import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
	path = 'C:/model.pb'
	#读取保存的文件，并将文件解析成对应的GraphDef Protocol Buffer
	with gfile.FastGFile(path,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromeString(f.read())
	result = tf.import_graph_def(graph_def,return_element=['add:0'])
	print sess.run(result)
```



