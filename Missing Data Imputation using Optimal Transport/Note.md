## Note

## Missing Data Imputation using Optimal Transport



### Contributions

- 应用了OT来定义损失函数，来做数据缺失的问题

  源自于一种直觉，在同一个数据集中的**随机batch**，应该服从**同一个分布**

- 提供了基于**OT距离损失函数**的数据填充的算法

  - non-parametric

    具有充分的自由度

    可以代表global shape，同时考虑local features

  - parametric models

    需要进行**迭代训练**

    可以与其他的填充策略进行结合

    - Multi-Layer Perceptrons

  

### Background

**missing data**

三种缺失

- Missing completely at random（MCAR）

  缺失独立于data

- missing at random（MAR）

  缺失的概率取决于观察到的数据

- missing not at random（MNAR）

  缺失的概率取决于未观察到的数据



经典的填充方法

- 依据联合分布

  假设数据为高斯分布，参数使用EM算法进行估计，缺省值从其预测的分布中得到

- 低秩结构，条件分布的方法



无参数的填充方法

- KNN填充方法
- 随机森林填充方法



**wasserstein distances, entropic regularization and sinkhorn divergences**

OT变得可以微分，并且可以用Sinkhorn iteration进行解决





### Imputing Missing Values using OT



