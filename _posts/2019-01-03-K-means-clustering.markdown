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
* Input: \\(\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N] \in \mathbb{R}^{d \times N}\\) data point and \\(K\\) number of clusters \\(K < N\\). 
* Output: \\(\mathbf{Y} = [y _ {i1}, y _ {i2}, \dots, y _ {iK}] \\) is label vector for each vector data point \\(\mathbf{x}_i\\). If \\(\mathbf{x}_i\\) is belong to to cluster \\(k\\) mean \\(\mathbf{y} _ {ik} = 1\\) and \\(y _ {ij} = 0, \forall j \neq k \\). This representation is call one-hot. We use this representation because two reasons: Firstly, K means clustering algorithm can not operate on label data directly, it should be numeric to calculate e.g the loss function below. Secondly, it works well in non-ordinal relationship problems e.g fruit problems. More detail [here](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/). 

\\[y_{ik} \in \{0, 1\},~~~ \sum_{k = 1}^K y_{ik} = 1 \\]
* Output: \\(\mathbf{M} = [\mathbf{m}_1, \mathbf{m}_2, \dots \mathbf{m}_K]\\) is the set of centroids (center of the cluster).
* Loss function(total distance of every point to every centroids)

\\[\mathcal{L}(\mathbf{Y}, \mathbf{M}) = \sum _ {i=1}^N \sum _ {j=1}^K y _ {ij} \|\mathbf{x}_i - \mathbf{m}_j\|_2^2\\]
* \\(\|\mathbf{x}_i - \mathbf{m}_k\|_2^2\\) is square of Euclidean distance, it hard to solve the derivative Euclidean function, so in the most case we take the power of two of this function. 
* \\(y _ {ik} \|\mathbf{x}_i - \mathbf{m}_k\|_2^2 =  \sum _ {j=1}^K y _ {ij} \|\mathbf{x} _ i - \mathbf{m} _ j\|_2^2 \\) : Above we have, \\(\mathbf{x}_i\\) belongs to cluster \\(k\\) then the label of \\(\mathbf{x}_i\\) in cluster \\(k\\) is 1 and 0 in other clusters, so the total distance of \\(\mathbf{x}_i\\) to every centroids is equal the distance of \\(\mathbf{x}_i\\) to the cluster it is assigned.
* Then, we have the loss function is the sum of all distance above. So, we have to minimize the loss function means: 
\\[\mathbf{Y}, \mathbf{M} = \arg\min_{\mathbf{Y}, \mathbf{M}} \sum _ {i=1}^N \sum _ {j=1}^K y _ {ij} \|\mathbf{x}_i - \mathbf{m}_j\|_2^2 \\]

\\[\text{subject to:} ~~ y_{ij} \in \{0, 1\}~~ \forall i, j;~~~ \sum_{j = 1}^K y_{ij} = 1~~\forall i\\]

* How we solve this problem? 
* The first step, suppose we have label \\(Y\\) to find the centroid \\(M\\): 
  * Task: Find the centroids
  * Experience: \\(N\\) data points, \\(K\\) cluster
  * Algorithm: Derivative
  * Performance: At each cluster, total distance of \\(N\\) data points to its centroid is minimum.
  * Function:
\\[\mathbf{m} _ j = \arg\min _ {\mathbf{m}_j} \sum _ {i = 1}^{N} y _ {ij} \|\mathbf{x}_i - \mathbf{m}_j \|_2^2.\\]
  We take the 2-sided derivative, then we can get the value (detail [here](https://machinelearningcoban.com/2017/01/01/kmeans/))
\\[\Rightarrow \mathbf{m} _ j = \frac{ \sum _ {i=1}^N y _ {ij} \mathbf{x}_i}{\sum _ {i=1}^N y _ {ij}}\\]
  
  => To get the minimum, the centroid should be the average of every point in its cluster. 

* The second step, suppose we have the centroid \\(M\\) to find the label \\(Y\\) (detail [here](https://machinelearningcoban.com/2017/01/01/kmeans/)):
  * Task: Find the label (how we assign data point to cluster)
  * Experience: \\(N\\) data points, \\(M\\) centroids
  * Algorithm: Compare Euclidean distance between each data points to each cluster, a data point is assigned to a cluster if the distance between it to the centroid is minimum when comparing to other centroids. 
  * Performance: the total distance of every data point is assigned to a cluster is minimal.
  * Function: 
\\[\mathbf{y} _ i = \arg\min _ {\mathbf{y}_i} \sum _ {j=1}^K y _ {ij}\|\mathbf{x} _ i - \mathbf{m}_j\|_2^2 \\]

  We haven't known the label, so every value of y is 1, so we can simplify the function as: 
  \\[j = \arg\min_{j} \|\mathbf{x}_i - \mathbf{m}_j\|_2^2\\]

### How it works in programming:
* In programming, we solve it by three steps:
  * Step 1: We choose k init centroid from the data set.
  * Step 2: Calculate the Euclidean distance to create the cluster (using the second way above)
  * Step 3: recalculate the centroids(using the first way)
  * => repeat step 2, 3 to until the after centroid is similar to the previous centroid. 
  * The complexity of this algorithm is O(n*k*t), n is number of object, k is number of cluster and t is iteration. More detail [here](https://www.researchgate.net/post/What_is_the_time_complexity_of_clustering_algorithms)

### Extension
Large data set: What happens if we have a data set with millions of object?
![mapreduce](../img/k-means-clustering-mapreduce.png)

  * Parallel K-means Clustering Based on Map Reduce
    * One of the basic and importance thing is the distance computation is independent, so we can slip the data set into multiple chunks for parallel execute. 
    * We use hadoop reprocess the big init data set.
    * Step 1: Mapper 
      * Input: \\(X = \begin{Bmatrix} x_1 , x_2 \ldots x_n \end{Bmatrix} \\) objects, \\(C = \begin{Bmatrix} c_1 , c_2 \ldots c_k \end{Bmatrix}\\) centroid
      * Output: List pair of \\((x_i, y_j)\\) with \\(1 \leq i \leq n\\) and \\(1 \leq j \leq k\\).
      * Algorithm: Assign nearest objects to the centroids. So, we back to problem that we have known the center of the cluster and assign nearest object to that cluster, using the second way that I have mentioned in mathematical. You can find the detail pseudocode [here](http://iip.ict.ac.cn/sites/default/files/publication/2009_Weizhong%20Zhao_Parallel%20K-means%20clustering%20based%20on%20mapreduce.pdf) 
    * Step 2: Combiner
      * Input: Output of the step 1
      * Output: \\(k\\) pair of \\((key, value)\\) with \\(key\\) in the centroids, and \\(value\\) is sum of values of object that is assigned to the cluster and number of objects.
      * Algorithm: combine by \\(key\\) (centroids)
      * Pseudocode: [here](http://iip.ict.ac.cn/sites/default/files/publication/2009_Weizhong%20Zhao_Parallel%20K-means%20clustering%20based%20on%20mapreduce.pdf) 
    * Step 3: Reducer
      * Input: Output of step 2
      * Output: New list of centroid
      * Algorithm: Now, it becomes to problem that we already know the assignment of object in each cluster, now find the new centroid. Using the first way in mathematical session. 
      * Pseudocode: [here](http://iip.ict.ac.cn/sites/default/files/publication/2009_Weizhong%20Zhao_Parallel%20K-means%20clustering%20based%20on%20mapreduce.pdf) 
      
Unknown \\(k\\): In case we don't have the \\(k\\), we can use elbow method to determine number of clusters in a data set.
* Input: \\(X = \begin{Bmatrix} x_1 , x_2 \ldots x_n \end{Bmatrix} \\) objects, \\(P_j = \begin{Bmatrix} x_i , x _ {i + 1} \ldots x_j \end{Bmatrix} \\) set of objects in cluster j, \\(M = \begin{Bmatrix} 1 , 2 \ldots k \end{Bmatrix}\\) the potential value for k
* Ouput: optimal \\(k\\) (the \\(k\\) is that the value of WCSS at this \\(k\\) will only change slightly, the elbow of the graph ).
* Algorithm: 
\\[\mathsf { WCSS } = \sum _ { { j = 1 } }^{i \in \mathsf { M }}\sum _ { { P_j }  \in \mathsf { Cluster } _j } \mathsf { distance } \left( { P } _ { { j } } , { C } _ { j } \right) ^ { 2 }\\] 
* Performance: Let draw an imaginary line with two value of \\(\mathsf { WCSS }\\). The elbow of the graph will be the point that has largest distance to the imaginary line.
* Function: \\(R ^ {m*n}\\) 
    ![](https://media.licdn.com/dms/image/C4E12AQGtYzxZcksIkQ/article-inline_image-shrink_1500_2232/0?e=1553731200&v=beta&t=aw7WXMvLTJxT6nw4o31o1UtwNkjV3A_oeip0q52rnWI)

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

  Center point: \\([2.99181004, 6.03338002], [9.01890293, 8.09573328], [1.97276695, 2.00616681], [8.00032512, 4.9943995 ]\\)
                   
![Result](../img/k-means-clustering-after.png)
*Result*

* Source code: [Here](https://github.com/HaiDang9719/StudyML/blob/master/K_means_clustering/k-meansClusteringex1.py)

* Application 

I hope you like it!

Source: 

[K means Clustering - ML cơ bản](https://machinelearningcoban.com/2017/01/01/kmeans/)

[K means Clustering - Complexity](https://www.researchgate.net/post/What_is_the_time_complexity_of_clustering_algorithms)

[K means Clustering - map reduce](http://iip.ict.ac.cn/sites/default/files/publication/2009_Weizhong%20Zhao_Parallel%20K-means%20clustering%20based%20on%20mapreduce.pdf)

[K means Clustering - map reduce example](https://pdfs.semanticscholar.org/46a3/d830f379ae61c269c9425d615f359067a5a6.pdf)

[Elbow method](https://www.linkedin.com/pulse/finding-optimal-number-clusters-k-means-through-elbow-asanka-perera/)
