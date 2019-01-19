---
layout: post
title: "K-nearest-neighbor"
image: knn2.jpg
author: Hai Dang
comments: true
---
# Welcome to my next blog
**Hello**, this is blog about K nearest neighbor.
### Why and When? 
* K nearest neighbor: we use this algorithm in classification and regression problems. Let make an example: 
  * Task: Classification fruit
  * Experience: Three kind of labeled fruit
  * Performance: Evaluate the accuracy of return guess value of the machine
  * Algorithm: K nearest neighbor
  * Function: \\(f(new fruit) -> I_f\\) : \\(I_f\\) means the label of fruit
* How it works? With this algorithm, the machine will try to guess the label of the new fruit based on the average of "voting" of the nearest neighbor of the new fruit. The nearer the neighbor is, the more value of the voting it has. As you can see, it does not require training process before, so we can call that is lazy learning. 

### Mathematical
* Input: \\(X\\) labeled data set, \\(k\\) number of nearest neighbors
* Output: 
  * Classification: class membership (predict label for test object)
  * Regression: the value for the object (mean value of nearest objects)
* Algorithm: Distance calculation and choose a good \\(k\\)
* Distance calculation:
  * Euclidean Distance (n-space):
  \\[\begin{aligned} d ( \mathbf { p } , \mathbf { q } ) = d ( \mathbf { q } , \mathbf { p } ) & = \sqrt { \left( q _ { 1 } - p _ { 1 } \right) ^ { 2 } + \left( q _ { 2 } - p _ { 2 } \right) ^ { 2 } + \cdots + \left( q _ { n } - p _ { n } \right) ^ { 2 } } \\ & = \sqrt { \sum _ { i = 1 } ^ { n } \left( q _ { i } - p _ { i } \right) ^ { 2 } } \end{aligned}\\]
  * Hamming Distance for categorical non-numeric attributes:
  \\[D _ { H } = \sum _ { i = 1 } ^ { k } \left| p _ { i } - q _ { i } \right| \\]
  \\[\begin{array} { l } { i f p = q \Rightarrow 0 } \\ { i f p \neq q \Rightarrow 1 } \end{array}\\]
  * Manhattan Distance (\\(p\\) and \\(q\\) are vectors in \\(n\\) space dimension):
  \\[d _ { 1 } ( \mathbf { p } , \mathbf { q } ) = \| \mathbf { p } - \mathbf { q } \| _ { 1 } = \sum _ { i = 1 } ^ { n } \left| p _ { i } - q _ { i } \right|\\]
  \\[\mathbf { p } = \left( p _ { 1 } , p _ { 2 } , \ldots , p _ { n } \right) \text { and } \mathbf { q } = \left( q _ { 1 } , q _ { 2 } , \ldots , q _ { n } \right)\\]
  * Minkowski Distance is the generalization of Euclidean and Manhattan distance:
    * \\(h\\) is nore, \\(h = 1\\) is Manhattan distance, \\(h = 2 \\) is Euclidean distance
  \\[d ( p , q ) = \sqrt [ n ] { \left| p _ { 1 } - q _ { 1 } \right| ^ { h } + \left| p _ { 2 } - q _ { 2 } \right| ^ { h } + \cdots + \left| p _ { n } - q _ { n } \right| ^ { h } }\\]
* how to find a good \\(k\\):
  * The value of \\(k\\) is too low or too high can lead to the high inaccuracy (over-fitting -> under-fitting)
  * One of possible way to choose k is \\(\mathbf(sqrt(n))\\) and should choose the odd value for \\(k\\), more explanation [here](https://www.quora.com/How-can-I-choose-the-best-K-in-KNN-K-nearest-neighbour-classification)
  

### Example
* Example1([Source code]()): 
  * Task: Classification iris flower
  * Experience: Iris data set from scikit-learn library
  * Performance: using function accuracy_store from scikit-learn library
  * Algorithm: KNN
  * Function: \\(f(x) -> X\\): \\(x\\) is unlabeled iris flower, \\(X\\) is labeled iris flower.
  * In this example, I try to using two method to improve the accuracy of the result: 
    * Using different \\(k\\)
  
{% highlight ruby %}  
for i in range(1,14):
    clf = neighbors.KNeighborsClassifier(n_neighbors = i, p = 2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("k value: ", i)
    print("Print results for 40 test data points:")
    print("Predicted labels: ", y_pred[10:50])
    print("Ground truth    : ", y_test[10:50])

    from sklearn.metrics import accuracy_score
    print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

{% endhighlight %}

   * using evaluating weight of voting, means the nearer one has the higher voting value.

{% highlight ruby %}  
for i in range(1,14):
    clf = neighbors.KNeighborsClassifier(n_neighbors = i, p = 2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("k value: ", i)
    print("Print results for 40 test data points:")
    print("Predicted labels: ", y_pred[10:50])
    print("Ground truth    : ", y_test[10:50])

    from sklearn.metrics import accuracy_score
    print("Accuracy: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

{% endhighlight %}
   * result: 
  
{% highlight ruby %}
k value:  1
Print results for 40 test data points:
Predicted labels:  [0 1 2 2 1 2 1 0 0 1 1 0 1 0 0 01 2 1 2 2 2 2 1 2 0 2 1 2 2 0 0 0 0 0 0 2
 1 2 1]
Ground truth    :  [0 1 2 2 1 2 1 0 0 1 2 0 1 0 0 01 2 1 2 2 2 2 1 2 0 2 1 2 2 0 0 0 0 0 0 2
 1 2 1]
Accuracy: 96.00 %
k value:  2
Print results for 40 test data points:
Predicted labels:  [0 1 2 2 1 2 1 0 0 1 1 0 1 0 0 01 2 1 2 1 2 2 1 1 0 2 1 2 2 0 0 0 0 0 0 2
 1 2 1]
Ground truth    :  [0 1 2 2 1 2 1 0 0 1 2 0 1 0 0 01 2 1 2 2 2 2 1 2 0 2 1 2 2 0 0 0 0 0 0 2
 1 2 1]
Accuracy: 92.00 %
k value:  3
Print results for 40 test data points:
Predicted labels:  [0 1 2 2 1 2 1 0 0 1 2 0 1 0 0 01 2 1 2 2 2 2 1 2 0 2 1 2 2 0 0 0 0 0 0 2
 1 2 1]
Ground truth    :  [0 1 2 2 1 2 1 0 0 1 2 0 1 0 0 01 2 1 2 2 2 2 1 2 0 2 1 2 2 0 0 0 0 0 0 2
 1 2 1]
Accuracy: 98.00 %
{% endhighlight %}

I hope you like it!

Source: 

[K nearest Neighbor - ML cơ bản](https://machinelearningcoban.com/2017/01/08/knn/)
     
