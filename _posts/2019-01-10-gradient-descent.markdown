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
    {% highlight ruby %} 
    def grad(x):
        return -2*np.sin(x)*np.sin(2*x)+ np.cos(x)*np.cos(2*x)+2*x
    
    def cost(x):
        return x**2 + np.cos(2*x)*np.sin(x)
    
    def myGD1(eta, x0):
        x = [x0]
        for it in range(1000):
            x_new = x[-1] - eta*grad(x[-1])
            if abs(grad(x_new)) < 1e-4:
                break
            x.append(x_new)
        return (x, it)
    (x1, it1) = myGD1(.01, -5)
    (x2, it2) = myGD1(.01, 5)
    {% endhighlight %}
    
   * result: 
    
    {% highlight ruby %} 
    Solution x1 = -0.272427, cost = -0.155893, obtained after 278 iterations
    Solution x2 = -0.272387, cost = -0.155893, obtained after 590 iterations
    {% endhighlight %}
    
* Gradient descent for multi-variable function
\\[\theta_{t+1} = \theta_{t} - \eta \nabla_{\theta} f(\theta_{t}) \\]
  * \\(\mathbf{\theta}\\) is a vector
  * Let make an example: 
    * Using gradient descent to find optimal value of the loss function of linear regression.
    * Source code: [Here](https://github.com/HaiDang9719/StudyML/blob/master/Gradient_descent/example2.py)
                       
![Result](../img/gradient-descent-ex2.png)

I hope you like it!

Source: 

[Gradient descent 1 - ML cơ bản](https://machinelearningcoban.com/2017/01/12/gradientdescent/)
