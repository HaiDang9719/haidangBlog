---
layout: post
title: "Gradient Descent"
image: gradient-descent-1.png
author: Hai Dang
comments: true
---
# Welcome to my next blog
**Hello**, this is blog about gradient descent.
### Why and When?
* Gradient descent is a method to find optimal point of the function. In machine learning, we try to get the optimal value for the functions such as loss function, ... For some complex function, it is really hard to find the global optimal value. So, one of the basic way is to divide the function into multiple part with much less complex and try to find the local optimal value of these part and compare together to find the quite global optimal one, that is how gradient descent works. 

### Mathematical
* Gradient descent for function of one variable:
  \\[x_{t+1} = x_{t} - \eta f’(x_{t})\\]
  * \\(n\\) is learning rate. It represents for the magnitude of step that the value of \\(x_{t}\\) get closer to \\(x^{*}\\) (the optimal value we need). Big learning rate make the process is faster but it will wildly oscillate around the minima without converging, otherwise it take a very long time to process. Choose the appropriate learning rate is very important. 
  * Let make an example: 
    * Function: \\(\sin ( x ) \cos ( 2 x ) + x ^ { 2 }\\)
    * Derivative Function: \\(- 2 \sin ( x ) \sin ( 2 x ) + \cos ( x ) \cos ( 2 x ) + 2 x \\)
    * We can make a simple python program to solve it: 
      * eta: learning rate
      * the program will show the result if the derivative function get the value lower than \\(0.0001\\). You can change the init value and learning rate to see the different. 
      * source code [here](https://github.com/HaiDang9719/StudyML/blob/master/Gradient_descent/example1.py)
    
* Gradient descent for multi-variable function
\\[\theta_{t+1} = \theta_{t} - \eta \nabla_{\theta} f(\theta_{t}) \\]
  * \\(\mathbf{\theta}\\) is a vector
  * \\(\nabla_{\theta} f(\theta_{t}) \\) is gradient vector, (steepest) direction in input space.
  * To ge better direction:
    * Consider the curve of approximation function (approximation at \\(x _ {0}\\) can be \\(f _ {0} \\), \\(f _ {1} \\), \\(f _ {2} \\) ), Jacobian, Hessian
    * Jacobian method: 
      * Input: \\(x = \left( x _ { 1 } , \ldots , x _ { j } , \ldots , x _ { n } \right) ^ { \top }\\)
      * Output: \\(f ( x ) = \left( f _ { 1 } ( x ) , \ldots , f _ { i } ( x ) , \ldots , f _ { m } ( x ) \right) ^ { \top }\\)
      \\[\left[ J _ { i j } : = \partial _ { j } f _ { i } ( x ) \equiv \frac { \partial } { \partial x _ { I } } f _ { i } ( x ) \right]\\]
  * Let make an example: 
    * Using gradient descent to find optimal value of the loss function of linear regression.
    * Source code: [Here](https://github.com/HaiDang9719/StudyML/blob/master/Gradient_descent/example2.py)

I hope you like it!

Source: 

[Gradient descent 1 - ML cơ bản](https://machinelearningcoban.com/2017/01/12/gradientdescent/)
