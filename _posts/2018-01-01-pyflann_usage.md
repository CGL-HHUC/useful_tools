---
layout:     post
title:      ""PyFlann 介绍"
subtitle:   "pyflann"
date:       2018-01-01
author:     Mcoder
catalog: true
---

# PyFlann 使用方法
`PyFlann` 其实是 `FLANN` 的 `python` 接口，当前支持python2 和 python3。`FLANN` 的意思是`Fast Library for Approximate Nearest Neighbors`，也就是快速解决最近点搜类问题的库。

这一类问题，是一个在尺度空间中寻找最近点的优化问题。问题描述如下：在尺度空间M中给定一个点集S和一个目标点q ∈ M，在S中找到距离q最近的点。很多情况下，M为多维的欧几里得空间，距离由欧几里得距离或曼哈顿距离决定。

最近点搜索问题的快速解决在很多领域都有着重要意义，如：图像识别及分类、机器学习、文档查重、统计学和大数据等。但是，当这个维度很高的时候，解决起来是一个相当困难的工作。这导致很多研究者开始对更好的解决这一问题产生了兴趣。这个库提供了`linear, kdtree, kmeans, composite, autotuned`几种算法来更好的解决问题。

# 安装

pip安装
```
pip install pyflann
```

源码安装
```
git clone https://github.com/primetang/pyflann.git
cd pyflann
[sudo] python setup.py install
```

# 使用
`pyflann` 包 提供了一个名为 FLANN 的类，来负责执行最近点搜索这个具体的操作。这个类包含如下的函数。

## def build_index(self, pts, `**`kwargs)
这个方法按照提交的算法，构建内部的数据结构排列，来加快后续的查找，如果需要多次查找的话，使用这种方法还是很棒的。它经常与方法 `def nn_index(self, qpts, num_neighbors = 1, **kwargs)` 共同使用。

pts 是数据集，必须是 `numpy` 的 `2D` 数组或 `matrix`，用 `row`优先方式存储。

`**`kwargs 是一组不定的参数，首先包含一个参数 `algorithm` ，然后根据 `algorithm` 参数的不同，后续的参数也是不同的，总共有如下几种情况。

```python
flann = pyflann.FLANN()

# 初始化 dataset
params = flann.build_index(dataset, algorithm = 'linear')
params = flann.build_index(dataset, algorithm = 'kdtree', trees)
params = flann.build_index(dataset, algorithm = 'autotuned',
    target_precision, build_weight, memory_weight, sample_fraction)
params = flann.build_index(dataset, algorithm = "means", branching,
    iterations, centers_init, cb_index)
params = flann.build_index(dataset, algorithm = "composite", tress, branching,
    iterations, centers_init, cb_index)
```

### linear
Linear 算法并没有创建内部index，它是采用了暴力法求解，线性查找，因此无其他参数，且速度非常慢。
调用函数如下:
```python
params = flann.build_index(dataset, algorithm = 'linear')
```

### autotuned
此算法使用 `Cross-Validation` 技术自动选择最好的 index 和操作参数。
需要手动填写:
- **target precision** - is a number between 0 and 1 specifying the percentage of the approximate nearest-neighbor searches that return the exact nearest- neighbor. Using a higher value for this parameter gives more accurate results, but the search takes longer. The optimum value usually depends on the application.

- **build weight** - species the importance of the index build time raported to the nearest-neighbor search time. In some applications it's acceptable for the index build step to take a long time if the subsequent searches in the index can be performed very fast. In other applications it's required that the index be build as fast as possible even if that leads to slightly longer search times. **(Default value: 0.01)**

- **memory weight** - is used to specify the tradeo between time (index build time and search time) and memory used by the index. A value less than 1 gives more importance to the time spent and a value greater than 1 gives more importance to the memory usage.

- **sample fraction** - is a number between 0 and 1 indicating what fraction of the dataset to use in the automatic parameter conguration algorithm. Running the algorithm on the full dataset gives the most accurate results, but for very large datasets can take longer than desired. In such case, using just a fraction of the data helps speeding up this algorithm, while still giving good approximations of the optimum parameters.

使用实例：
```python

from pyflann import *
from numpy import *
from numpy.random import *

dataset = rand(10000, 128)
testset = rand(1000, 128)

flann = FLANN()
params = flann.build_index(dataset, algorithm="autotuned", target_precision=0.9, log_level = "info");
print params

result, dists = flann.nn_index(testset,5, checks=params["checks"]);
```
其中， `result` 是一个 `numpy.ndarray` ，shape 为： 1000 x 5， 及与测试集相关，与查询的最近点个数 k 相关。

### kd tree
[k-d tree](https://www.cnblogs.com/eyeszjwang/articles/2429382.html)，是一种分割k维数据空间的数据结构。主要应用于多维空间关键数据的搜索（如：范围搜索和最近邻搜索）。K-D树是二进制空间分割树的特殊的情况。

这种算法需要提供一个参数`trees` ，建议填写的数值是 `4`。
调用过程为：
```python
params = flann.build_index(dataset, algorithm = 'kdtree', trees=4)
```

### kmeans
Hierarchical k-means 算法。需要输入以下参数：

- **branching** - the branching factor to use for the hierarchical kmeans tree creation. While kdtree is always a binary tree, each node in the kmeans tree may have several branches depending on the value of this parameter.

- **iterations** - the maximum number of iterations to use in the kmeans clustering stage when building the kmeans tree. A value of -1 used here means that the kmeans clustering should be performed until convergence.

- **centers_init** - the algorithm to use for selecting the initial centers when performing a kmeans clustering step. The possible values are 'random' (picks the initial cluster centers randomly), 'gonzales' (picks the initial centers using the Gonzales algorithm) and 'kmeanspp' (picks the initial centers using the algorithm suggested in [AV07]). If this parameters is omitted, the default value is 'random'.

- **cb_index** - this parameter (cluster boundary index) in uences the way exploration is performed in the hierarchical kmeans tree. When cb index is zero the next kmeans domain to be explored is choosen to be the one with the closest center. A value greater then zero also takes into account the size of the domain.

### composite
是将 `kmeans` 和 `kdtree` 算法混合使用，需要将两者的参数都提供。

## def nn_index(self, qpts, num_neighbors = 1, `**`kwargs)
这个函数是执行完成 `build_index` 函数之后进行查询最近点时调用的函数。

- qpts: 待查询的 testset，维度要与之前建立时使用的数据集一样。比如建立的数据集是 1000 x 3的矩阵，查询的数据集必须为 n x 3 的矩阵，否则无法进行查询。
- num_neighbors: 查询最近的几个点，根据这个值决定返回的数值。如果 testset 的为 1000 x 3 矩阵，查询最近的五个节点，则返回 1000 x 5的矩阵，如果查询最近的一个节点，则返回 1000 x 1 的数组。
- kwargs: checks=["checks"]

## def nn(self, pts, qpts, num_neighbors = 1, `**`kwargs)
这个函数是将 `build_index` 与 `nn_index` 合并为一，适合只执行一次的查询，如果需要对一组数据进行多次查询的话并不建议使用。

```python
from pyflann import *
import numpy as np

dataset = np.array(
    [[1., 1, 1, 2, 3],
     [10, 10, 10, 3, 2],
     [100, 100, 2, 30, 1]
     ])
testset = np.array(
    [[1., 1, 1, 1, 1],
     [90, 90, 10, 10, 1]
     ])
flann = FLANN()
result, dists = flann.nn(
    dataset, testset, 2, algorithm="kmeans", branching=32, iterations=7, checks=16)
print result
print dists
```

## def save_index(self, filename)
保存下建立的数据集索引，但不保存数据集。因此使用时，仍需要手动加载原数据集配合查询到的 index 使用。

## def load_index(self, filename, pts)
加载保存的索引，需要记得自己要另外加载数据集。

## def set_distance_type(distance type, order = 0)
这个函数设置了计算使用的距离类型。

- **type** - the distance type to use. Possible values are: 'euclidean', 'manhattan', 'minkowski', 'max dist' (L infinity - distance type is not valid for kd-tree index type since it's not dimensionwise additive), 'hik' (histogram intersection kernel), 'hellinger','cs' (chi-square) and 'kl' (Kullback-Leibler).

- **order** - only used if distance type is 'minkowski' and represents the order of the minkowski distance.

# Reference
- [FLANN](https://www.cs.ubc.ca/research/flann/)
- [FLANN User Guided](http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf)
- [K-d Tree](https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/kdtrees.pdf)
