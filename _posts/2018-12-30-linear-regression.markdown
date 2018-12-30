---
layout: post
title:  "Linear Regression"
image: ML1.jpg
author: Hai Dang
comments: true
---

# Welcome to my first blog

**Hello**, this is my first blog post about linear regression in machine learning series.
### Semantic meaning of linear regression
Let make an example first, we have a house and we need the machine to estimate the price of the house. So, what will be the important attribute we need to concern? Obviously, there is a list of attributes:
* The position of the house, is it near the city center or supermarket? 
* The size of the house, is it big or small?
* The number of rooms
* .....

Those information is presented as a vector to input to the machine, then we give them the sample price pair with the vector of information. As a result, when we input any block of information about the house, we expect the machine will return the expected price with minimal difference between it and the real price. To achieve that, we need to understand what actually happens inside the machine? So, we come to first and basic technique that happens inside the machine, linear regression. 

To get the expected price of the house, we actually need to know the influence of the attributes to the price of the house and this relationship is represented by linear regression. Why it is called linear regression? Because we represent it into a space as a line for 2D space and a hyperplane for 3D space. In machine learning, linear is most basic and common model, so it is a good starting point for study machine learning. 

### Mathematical 

In this part, I would like to summary the formula of linear regression, for more detail, you can read at link in source part.

In linear regression, the relationship is represented a linear function or linear hyperplane and the result of the function is the prediction value:

 2D space: \\[ y \approx \mathbf{\bar{x}}\mathbf{w} = \hat{y} \\]
 
 We expect the predict price has as small difference as possible when comparing to the real price. The difference or we can call as loss is represented as the following function:
 
\\[\mathcal{L}(\mathbf{w}) = \frac{1}{2}\sum_{i=1}^N (y_i - \mathbf{\bar{x}_i}\mathbf{w})^2  \\]
To reduce the difference between predicted and real value, we find the value w so that the loss function has the minimum value.

\\[\mathbf{w}^* = \arg\min_{\mathbf{w}} \mathcal{L}(\mathbf{w})  \\]

To find the minimum value, one of the popular way is use derivative. 
\\[\frac{\partial{\mathcal{L}(\mathbf{w})}}{\partial{\mathbf{w}}} 
   = \mathbf{\bar{X}}^T(\mathbf{\bar{X}}\mathbf{w} - \mathbf{y})  \\]
and the above function is equal 0 when it reaches the minimum value of w

\\[\mathbf{\bar{X}}^T\mathbf{\bar{X}}\mathbf{w} = \mathbf{\bar{X}}^T\mathbf{y} \triangleq \mathbf{b}  \\]

So, as a conclusion, the optimal point of linear regression is: 

\\[\mathbf{w} = \mathbf{A}^{\dagger}\mathbf{b} = (\mathbf{\bar{X}}^T\mathbf{\bar{X}})^{\dagger} \mathbf{\bar{X}}^T\mathbf{y}  \\]

Now, let come to an example to get better understand how we solve the problem with the above function.

We have a data for both training and test set: 
![](../img/dataset-linear-regression.jpg)
And the problem: 
In this problem you will implement the linear basis method for regression with least squares error. You will estimate functions using the training data and evaluate their performance on the test data. You should solve this problem by setting up a linear system of equations Aw = b, where the solution to the linear equations minimizes the sum of squared errors.
For this problem you should use polynomial basis functions:
\\[\phi _ { 1 } ( x ) = 1 , \phi _ { 2 } ( x ) = x , \phi _ { 3 } ( x ) = x ^ { 2 } , \phi _ { 4 } ( x ) = x ^ { 3 } , \ldots \\]
\\[f _ { w } ( x ) = \sum _ { i = 1 } ^ { M } w _ { i } \phi _ { i } ( x )  \\]
Let T: 
\\[T = { \left( x _ { 1 } , y _ { 1 } \right) , \ldots , \left( x _ { N } , y _ { N } \right)\}  \\]
And RMS is: 
\\[E ( F , T ) = \sqrt { \frac { \sum _ { i = 1 } ^ { N } \left( f _ { w } \left( x _ { i } \right) - y _ { i } \right) ^ { 2 } } { N } }  \\]
Let begin:

Degree 1: We firstly create the matrix H1(combine 1 and X value in the training set), then calculate the transpose of matrix H1. Then, we multiple the matrix H1 with the transpose matrix of H1 to get the A value. 
For b value, we multiple the transpose matrix of H1 with t value( in this case is the matrix of Y value in the training set. 
To calculate w*: We multiple the inverse value of A with value of b.
And the result has two value(a,b) present for the linear function:  f(x) = ax + b.

![](../img/linear-regression-ex1.png)

After we have w value for each degree, we apply it to the above formula to calculate the RMS.
With training set, we calculate the sum of the power-two of the substraction beween polinomial function and Y. Then, we take square root of the sum over the number of data point. The detail of calculation is included in the image below.
![](../img/linear-regression-ex1-rms-training.png)
For test set: 
![](../img/linear-regression-ex1-rms-test.png)

For more detail, you can download source code (python) here: [Link](https://drive.google.com/file/d/1fCnqYUKJVz8_ziw7cnVTpMx96o-Hkrzn/view?usp=sharing)

Python also provide a library scikit-learn that have linear model. So, you can use this library to test the result after finishing the exercise.

You can also practice with the example in [Linear Regression - ML cơ bản](https://machinelearningcoban.com/2016/12/28/linearregression/). 

I hope you like it!

Source: 

[Linear Regression - ML cơ bản](https://machinelearningcoban.com/2016/12/28/linearregression/)

