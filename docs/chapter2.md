## TF环境搭建

1.TF主要依赖的包

**Protocol Buffer**

是谷歌开发的处理结构化数据的工具，Protocol Buffer序列化（将结构化的数据变成数据流的格式，白话就是变成一个字符串）后得到的数据是不可读的字符串，而是二进制流。如何从结构化的数据序列化，从序列化之后的数据流中还原原来的数据结构，这是Protocol Buffer解决的主要问题。是Tensorflow系统用到的重要工具。

**Bazel**

Bazel是从谷歌开源的自动化构建工具，谷歌内部大部分的应用都是通过它来**编译**的，Tensorflow也是通过Bazel编译的。

2.TF安装

TensorFlow提供了多种不同的安装方式，我们主要介绍通过Docker安装，通过Pip安装及从源码安装。

**Docker安装**

Docker是新一代的虚拟化技术，可以将TF及TF的所有依赖关系封装在Docker镜像中，大大简化安装过程（我的Tensorflow就是通过Docker安装的，当时Tensorflow还不能很好的支持Windows)。那首先需要安装Docker(Docker的安装和使用不是本节的重点)，安装好后使用一个打包好的Docker镜像，TensorFlow官方提供了多个Docker镜像，国内的（eg才云科技）也提供了相关的镜像

```
docker run -it -p 8888:8888 -p 6006:6006
#启动一个TensorFlow容器
```
**Pip安装**

前期时pip在Windows下是无法安装的，现在可以在python3的环境下直接`pip install tensorflow`即可完成安装。极力推荐这种安装方式

**源码编译安装**

这种安装方式不建议大家采用，因为TensorFlow依赖于其他包，你需要提前把其他依赖包安装好之后才可正常通过源码安装TF。

安装好后测试一下你的TensorFlow能否正常使用,如果正常使用，恭喜你安装成功。

```python
import tensorflow as tf
a = tf.constant([1.0,2.0],name='a')
b = tf.constant([2.0,3.0],name='b')
result =a + b

sess = tf.session()
sess.run(result)
sess.close()
```
----

