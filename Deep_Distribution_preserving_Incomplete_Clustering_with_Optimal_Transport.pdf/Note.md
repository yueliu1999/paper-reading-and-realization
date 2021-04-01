## Note

### abstract

两点工作

1. 分布距离从像素的loss function被OT所代替，为了重建
2. 聚类的latent feature loss被用来regularize embedding



### Introduction

聚类的方法

- K-means
- Gaussian Mixture Model
- spectral clustering
- deep clustering



但是数据不完全非常可能发生，比如

- sensor failure
- unfinished collection
- data storage corruption



现有的不完全聚类可以分为两种机制

- heuristic-based
- learning-based

首先impute缺失的特征，可以用传统的聚类算法

- 启发式的方法

  使用一些统计属性

  - zero-filling
  - mean-filling
  - median value
  - KNN-filling

  当数据量很大的时候，效果很差，因为无法捕捉到更多的信息

- 学习的方法

  - shallow

    通常认为数据是low-rank的

    - EM算法

  - deep learning framework

    - GAN

    - VAE




现有方法的缺点

- imputation和clustering是分离开的
- 对于高维数据，不work



提出了一个**Deep Distribution-preserving Incomplete Clustering with Optimal Transport**

在Deep Embedding Clustering network 处理消失的特征

和现有的不同的是，最小化Wasserstrain distance between observed data and reconstructed data



可以同时imputation和embedded clustering procedures



贡献：

1. 提出了一种noval end-to-end deep clustering network，最小化wasserstrain distance
2. regular the latent distribution，可以提高性能
3. sota



### Notion and Related Work



**Related work**

- Statistical imputation

  - zero
  - mean
  - median
  - KNN
  - Bayesian, EM

  

- Deep incomplete clustering

  一般都是分为two step

  - imputation
  - clustering

  有以下几个方法

  - GAIN，with GAN

    the discriminator in GAIN是为了准确的判别数据是填充的还是真实的

    缺点是不太稳定，不好训练和优化

  - VAEAC，based on variational antoencoder

  - Markov chain Monte Carlo(MCMC)

    

- Optimal transport and sinkhorn divergence

  给定两个离散的概率分布

  $\alpha = \sum_{i=1}^na_i\delta_{X_i}$

  $\beta = \sum_{i=1}^nb_i\delta_{Y_i}$

  q-th Wasserstein distance corresponds to these two distribution $\alpha$ and $\beta$

  $W_q(\alpha,\beta) = \min\limits_{P\in U(a,b)}<F,c>$

  $C = (||x_i-y_j||^q)$是cost矩阵

  set q=2

  加上一个entropy regularization

  $W_q^{\epsilon}(\alpha,\beta) = \min\limits_{F\in U(a,b)}<F,C>-\epsilon h(F)$

  $h(F) = -\sum_{ij}f_{ij}logf_{ij}$是entropy regularization

  可以用sinkhorn算法来求解，得到一个对称的散度

  $S_{\epsilon}(\alpha,\beta) = OT_{\epsilon}(\alpha, \beta)-\frac{1}{2}(OT_{\epsilon}(\alpha,\alpha)+OT_{\epsilon}(\beta,\beta))$

  使用sinkhorn divergence来度量OT distance of two distributions

  

### DDIC-OT

**motivation**

**problem analysis**

前人的方法只能针对低维度的数据，无法针对高维度的数据

定理1：

假设数据是独立同分布的，当面临缺失的时候，一个完全观测的高维度数据以低概率存在

证明1：



**overall Network architecture**



clustering model

- encoder
- decoder
- soft clustering layer



loss是两个损失的线性相加，联合优化

$L = L_s + \gamma L_c$

其中$L_s$是sinkhorn divergence，$L_c$是clustering loss，$\gamma$是超参数

$S_{\epsilon}(\alpha,\beta) = OT_{\epsilon}(\alpha, \beta)-\frac{1}{2}(OT_{\epsilon}(\alpha,\alpha)+OT_{\epsilon}(\beta,\beta))$



P是soft assignment of the distribution z:

$p_{ij} = \frac{(1+||z_i-u_j||^2)^{-1}}{\sum_j(1+||z_i-u_j||^2)^{-1}}$

$q_{ij} = \frac{p_{ij}^2/\sum_ip_{ij}}{\sum_j(p_{ij}^2/\sum_ip_{ij})}$



clustering loss也就是

$L_c = KL(Q||P) = \sum\limits_i\sum\limits_jq_{ij}log\frac{q_{ij}}{p_{ij}}$



总结优点

1. $L_s$loss完成了重建任务，并且保留了几何特征
2. 更加flexible
3. 加入了正则



**model training**

- pre-training phase
- fine-tune phase
  - optimal transport distance
  - custering loss

encoder:d-500-500-1000/2000-10

decoder:10-1000/2000-500-500-d



Adam, lr=0.001

256 batch size

sinkhorn

$\epsilon = 0.01$ $\delta = 0.1$ $MaxIter = 200$



DDIC-OT算法流程

输入：

- 缺失的数据$X_m$
- 聚类数量k
- 超参数$\lambda$
- Batchsize N
- 最大迭代次数MaxIter
- stopping threshold $\delta$
- 学习率

输出：聚类assignmentS



步骤：

初始化$X_m$用mean-filling

初始化聚类中心u

- for iter=0 to Maxiter do

  - for i=0 to n/N do

    sample a minibatch from X

    计算相关的变量$z_i = f_e(x_i), \hat x_i = f_d(z_i)$

    计算$P_i$和$Q_i$

    计算clustering assignment for X

    计算overall1 loss L

    反向传递并更新模型参数

  - end for

    计算Z = $f_e(X)$

    计算Q和P

    计算聚类中心assignment

    if <$\delta$

    ​	停止训练







### Expiment



#### Dataset

- MNIST-full and Fashion-MNIST
- USPS
- COIL-20
- REuters-10K
- Letters



integrity ratios, means the percentage of missing features in all samples



#### Evaluation Metrics

- ACC

- NMI

  the normalised mutual dependence between the predicted labels and the ground-truth

- purity

  nums of sample correctly clustered / total nums of samples

  

















