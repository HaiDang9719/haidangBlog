---
layout: post
title: "K-means-clustering"
image: knn-1.png
author: Hai Dang
comments: true
---
# Welcome to my next blog
**Hello**, this is blog about K means clustering.
### Why and When? 
* K means clustering: We usually use this algorithm for unsupervised learning problem. With the unlabeled data, e.g we have multiple images of apple and mango, but we don't know which one is apple and which one is mango. So, basically, we will create two clusters based on size, shape, ... for these two kind of fruit and decide the most typical one will be the center of each cluster -> this is K means clustering.

### Mathematical 
* \\(\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N] \in \mathbb{R}^{d \times N}\\) data point, \\(K < N\\) is number of clusters. 
* \\(\mathbf{y}_i = [y _ {i1}, y _ {i2}, \dots, y _ {iK}] \\) is label vector for each vector data point \\(\mathbf{x}_i\\). If \\(\mathbf{x}_i\\) is belong to to cluster \\(K\\) mean \\(\mathbf{y} _ {ik} = 1\\) and \\(y _ {ij} = 0, \forall j \neq k \\). This representation is call one-hot. 

\\[y_{ik} \in \{0, 1\},~~~ \sum_{k = 1}^K y_{ik} = 1 \\]
* Why we need to use this representation and when we use it?
  * K means clustering algorithm can not operate on label data directly, it should be numeric to calculate e.g the loss function below. 
  * To transform categorical to numeric form, there are two way to convert: Integer Encoding and One-hot Encoding. There is a limitation of using Integer Encoding is that it works well in case the ordinal relationships exits in the problem, otherwise, it will create a mistake. More detail, you can further read [here](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/). 
* Loss function: 

\\[\mathcal{L}(\mathbf{Y}, \mathbf{M}) = \sum _ {i=1}^N \sum _ {j=1}^K y _ {ij} \|\mathbf{x}_i - \mathbf{m}_j\|_2^2\\]

* \\(\mathbf{Y} = [\mathbf{y}_1; \mathbf{y}_2; \dots; \mathbf{y}_N]\\) is lable vector and \\(\mathbf{M} = [\mathbf{m}_1, \mathbf{m}_2, \dots \mathbf{m}_K]\\) is center of cluster. So, to get the better result, we need to optimize 
  
\\[\mathbf{Y}, \mathbf{M} = \arg\min_{\mathbf{Y}, \mathbf{M}} \sum _ {i=1}^N \sum _ {j=1}^K y _ {ij} \|\mathbf{x}_i - \mathbf{m}_j\|_2^2 \\]

\\[\text{subject to:} ~~ y_{ij} \in \{0, 1\}~~ \forall i, j;~~~ \sum_{j = 1}^K y_{ij} = 1~~\forall i\\]

* \\(Y, M\\) are the variables need to find to reach the optimal value of the above function. So, we have two ways: 
* The first way, fix \\(Y\\) to find \\(M\\):
\\[\mathbf{m} _ j = \arg\min _ {\mathbf{m}_j} \sum _ {i = 1}^{N} y _ {ij} \|\mathbf{x}_i - \mathbf{m}_j \|_2^2.\\]
 
  We take the 2-sided derivative, then we can get the value (detail [here](https://machinelearningcoban.com/2017/01/01/kmeans/))
\\[\Rightarrow \mathbf{m} _ j = \frac{ \sum _ {i=1}^N y _ {ij} \mathbf{x}_i}{\sum _ {i=1}^N y _ {ij}}\\]

* The second way, fix \\(M\\) to find \\(Y\\) (detail [here](https://machinelearningcoban.com/2017/01/01/kmeans/)):
\\[\mathbf{y} _ i = \arg\min _ {\mathbf{y}_i} \sum _ {j=1}^K y _ {ij}\|\mathbf{x} _ i - \mathbf{m}_j\|_2^2 \\]

### Example
5-tuples: T, E, P, A, F
* Task: Find the center of each clusters
* Experience: 1500 points and 3 clusters
* Performance: compare the result with expected center we have created from the beginning.
* Algorithm: K means clustering
* Function: \\((a, b) = f(A)\\) \\((a, b)\\) is the coordinate of the center of cluster \\(A\\).
* You can change the coordinate of test center point and increase the number of point in each cluster as well as number of cluster to check the accuracy.
* The result also depend on separation between these clusters, if the clusters are quite mixed, the result will have a significant error. 
       
  Center point: \\([2, 2], [8, 5], [3, 6], [9,8]\\)
      
![](../img/k-means-clustering.png)
*Input data*

  Center point: \\([2.99181004 6.03338002], [9.01890293 8.09573328], [1.97276695 2.00616681], [8.00032512 4.9943995 ]\\)
                   
![Result](../img/k-means-clustering-after.png)
*Result*

* Source code: [Here](https://github.com/HaiDang9719/StudyML/blob/master/K_means_clustering/k-meansClusteringex1.py)

* Application 

I hope you like it!

Source: 

[K means Clustering - ML cơ bản](https://machinelearningcoban.com/2017/01/01/kmeans/)
