---
layout: post
title: "Introduction to AI and Machine Learning - Part1"
image: introduction-to-ml.png
author: Hai Dang
comments: true
---
# Welcome to the introduction series about AI and Machine Learning - Part1

**Hello**, machine learning and AI is one of "hot" topic in computer science nowadays. So, in this series, I would like to summary the basic concept and mathematics on this topic. 
1. What is this?
 * Artificial Intelligence: the ability of the machine that can simulate one of human intelligence abilities, e.g if we input a fish image, the machine knows this is a fish. 
 * Machine Learning: the learning process of the machine to improve its performance through experience. To get the better understand, you can look at example of how the machine can recover colorful image from black-white image [here](https://www.facebook.com/curiousAI/videos/1431500110327587/?fref=mentions). 
2. General model of Machine Learning
* T(Task): what we want the machine do
* E(Experience): input data set
* P(Performance): evaluation method 
* A(Algorithm): algorithm to find the optimization function in function space
* F(Function space): function space for machine works

   -> Machine Learning is process of computer to find automatically the best optimization function in function space. So, in the next part, we will focus on function, function space and how we represent a function space as well as how we can find an optimization function in that space.
3. Space of multi-variable functions and Gradients
   1. Function
   * Affine function \\[\mathbf{f(x)} = \mathbf{a}x+ \mathbf{b},  x \in \mathbb { R } \\] 
   * Quadratic function \\[\mathbf{f(x)} = \mathbf{a}x^2+ \mathbf{b}x +c, a \ne 0, a,b,c \in \mathbb { R } \\] 
   * We have two way to represent function on 2D space:
     * The first way, with the value of x, we can find the corresponding y by the function. The graph of the function is the set of those pair of x and y. 
     * The second way, with the value of x, we keep the value of y constant and change the parameters of the function, it is called as level sets/curves/contour lines/ map. 
     * Function space is set of function with different value of parameters. For example of affine function,  the change of value of **a** leads to the change of slope and the change of value of **b** to translate the function. 
   * After we have the function space, we need to find the value of parameters in the function space so that the function achieves the optimal value (peak or valley) by measure the direction, magnitude of the function,the distance between different functions and the direction that the function grows fastest. 
4. Vector and vector space
   
I hope you like it!

Source: 

[Machine Learning - Lecture 1](https://www.facebook.com/curiousAI/videos/1431500110327587/?fref=mentions)
[Machine Learning - Lecture 2](https://www.facebook.com/curiousAI/videos/1443931999084398/?hc_location=ufi)
 
