## Optimal Transport

Sinkhorn Distances: Lightspeed Computation of Optimal Transport



Pure math->Applied math->Computational math->machine learning applications



Monge问题：

- 将一堆沙子高效运输或reshape到另外一个目的地

目标：

建立描述**两个概率分布**的**几何的工具**





### 最优传输的基本概念

直方图或者概率向量a

$a\in\sum_n:=[a\in R_+^n | \sum_{i=1}^na_i = 1]$

离散的测度，discrete measures

$\alpha = \sum_{i=1}^na_i \delta_{x_i}$



最优指派问题：



monge问题：





**最优传输问题：**

![1](.\pic\1.png)

![2](.\pic\2.png)

**最优传输问题目标函数**

$L_C(a,b) = \min\limits_{P\in U(a,b)}[<C,P> = \sum\limits_{i,j}C_{i,j}P(i,j)]$

Cost矩阵和策略矩阵

线性规划问题，解不唯一



**Optimal Transport的度量**

假设m==n，给定参数p>=1 $C = D^p$

D矩阵的性质

- D是对称的
- D对角线为0
- 满足三角不等式, $D_{i,k}<=D_{i,j}+D_{j,k}$

p-Wasserstein distance

$W_p(a,b) = L_{D^P}(a,b)^{\frac{1}{p}}$

一般p=1，也就是说**cost矩阵C**就是**距离D**



**Wasserstein Barycenter**

以wasserstein distance为度量的中心







### 应用

- Machine Learning

- Computer vision and graphics
- robust optimization



Label Distribution Learning

是multi-label learning的拓展，主要关心label之间的相对关系

训练数据：

$[(x_i,y_i)]_{i=1}^m$



Wasserstein Generative Adversarial Network(WGAN)



Zero-shot Learning

有效的将模型学习到的知识转换到模型没见过的类





### 如何计算OT

加入entropic regularization

$L_C^{\epsilon}(a,b) = \min\limits_{P\in U(a,b)}<C,P> - \epsilon H(P)$

计算P矩阵到K矩阵的KL散度

$K_{i,j} = exp(-\frac{C_{i,j}}{\epsilon})$

$P_{ij} = u_iK_{ij}v_j$

$u_i = e^{f_i/\epsilon}$

$v_i = e^{g_j/\epsilon}$

![3](.\pic\3.png)



sinkhorn算法：

![4](.\pic\4.png)





### Example: Inexact PPA and WGAN

PPA算法，不用正则的OT，直接求解OT问题

因为引入了$\epsilon$所以会变得敏感，结果不是特别好

![5](.\pic\5.png)





Wassersteian GAN

最优化Wassersteian distance

用了不同的求解OT问题的方法





### Discussion



