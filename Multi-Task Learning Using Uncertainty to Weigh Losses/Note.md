## Note



### Abstract

许多**回归**和**分类**任务



给多个loss function进行加权，使用**homoscedastic uncentainty of each task**

- per-pixel depth regression
- semantic and instance segmentation from a monocular input image



### Introduction 

目标是为了提升

- 效率
- 准确率

**从共享表示中学习多个目标函数**



以前的方法：

直接将loss简单加权

- uniform
- 手动调节参数



每个任务的最优参数选择，是取决于

- measurement scale
- 任务噪音的大小



提出了一种principled方法通过结合多种loss，同时学习多个权重，通过**方差不确定性**，将方差不确定性解释为**任务依赖加权**，并展示了如何如何学会**平衡回归和分类损失**，可以最优的平衡这些损失，从而获得更好的性能



contribution

- 使用**方差不确定性**进行结合分类和回归的问题的loss
- 一种统一的语义分割、实例分割、深度回归架构
- 展示了loss权重的重要性在多任务网络中，并且如何获取最高的性能





### Related work

使用一种共享的表示，从多个任务中学习，其中一个任务可以帮助到其他的任务

其他人一般都是

- 简单的加权和
- 权重是平均的或者手工调节的



### Multi Task Learning with Homoscedastic Uncertainty

一般都是线性相加

$L_{total} = \sum_i w_iL_i$



ideas from probabilistic modelling



#### Homoscedastic uncertainty as task-dependent uncertainty 

在贝叶斯建模中，有两个不确定性可以进行建模

- 认知不确定性，捕获模型不知道的，因为缺少训练数据，

  可以通过增加训练数据来进行解释

- 偶然的不确定性，捕获到我们关于我们的数据无法解释信息的不确定性

  - 数据相关，取决于输入数据，被预测为模型输出
  - 任务相关，不依赖于输入数据



在多任务的setting中，捕获任务之间的**相对置信度**，反映回归或分类任务固有的不确定性，取决于

- 任务的表示
- unit of measure

使用了同方差不确定性作为基础



#### multi-task likeihoods

基于同方差不确定性高斯似然的多任务损失函数

似然函数

$f^W(x)$为神经网络的输出

回归问题：

$p(y|f^W(x)) = N(f^W(x), \delta^2)$

分类问题：

$p(y|f^W(x)) = Softmax(f^W(x))$



模型多个输出，定义了多任务的似然

![image-20210319220649187](C:\Users\liuyue\AppData\Roaming\Typora\typora-user-images\image-20210319220649187.png)



$L(W,\delta_1,\delta_2) = \frac{1}{2\delta_1^2}L1(W)+\frac{1}{2\delta_2^2}L_2(W)+log\delta_1\delta_2$



- 当参数的噪声**变大的时候，参数的权重就减小**
- 在最后一项中，噪声不会太大，因为是正则化









