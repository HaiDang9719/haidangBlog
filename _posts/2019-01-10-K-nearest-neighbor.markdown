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
* How it works? With this algorithm, the machine will try to guess the label of the new fruit based on the average of "voting" of the nearest neighbor of the new fruit. The nearer the neighbor is, the more value of the voting it has. As you can see, there is no learning in here, so we can call that is lazy learning. 

### Mathematical
* In this algorithm, the most important thing is how to find a good \\(k\\). A bad \\(k\\) can lead to a bad result. 
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
     
