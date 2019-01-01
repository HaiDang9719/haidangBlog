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
   * Affine function \\[f ( x ) = \mathbf{a}x+ \mathbf{b},  x \in \mathbb { R } \\] 
   * Quadratic function \\[f ( x ) = \mathbf{a}x^2+ \mathbf{b}x +c, a \ne 0, a,b,c \in \mathbb { R } \\] 
   * [Logistic sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)
   \\[f ( x ) = \frac { 1 } { 1 + e ^ { - x } } \\] 
   * [Hyperbolic tangent](https://en.wikipedia.org/wiki/Sigmoid_function)
   \\[f ( x ) = \tanh x = \frac { e ^ { x } - e ^ { - x } } { e ^ { x } + e ^ { - x } } \\] 
   * We have two way to represent function on 2D space:
     * The first way, with the value of x, we can find the corresponding y by the function. The graph of the function is the set of those pair of x and y. 
     * The second way, with the value of x, we keep the value of y constant and change the parameters of the function, it is called as level sets/curves/contour lines/ map. 
     * Function space is set of function with different value of parameters. For example of affine function,  the change of value of **a** leads to the change of slope and the change of value of **b** to translate the function. 
   * After we have the function space, we need to find the value of parameters in the function space so that the function achieves the optimal value (peak or valley) by measure the direction, magnitude of the function,the distance between different functions and the direction that the function grows fastest. 
4. Vector and vector space

   4.1. Vector
   * Vector is an object that has both direction and magnitude, it can be represented as a directed line segment and as a point in vector space. 
   ![color: white](../img/introduction-to-ml-vector.png)
   
   4.2. [Vector space](http://www.math.toronto.edu/gscott/WhatVS.pdf)
   * Vector space is a set of vector that satisfy two conditions of two operation:
     * An operation called vector addition that takes two vectors v, w ∈ V , and produces a third vector, written v + w ∈ V .
     * An operation called scalar multiplication that takes a scalar c ∈ F and a vector v ∈ V , and produces a new vector, written cv ∈ V .
     
   4.3. Meaning
   * In machine learning, each vector contains multiple parts and each part represent for a characteristic of an object. These parts connect together by Cartesian product to create a coordinate space.
   \\[V = \mathbb { R }^n = \mathbb { R } \times \mathbb { R }...\times\mathbb { R } : Cartesian-product\\] 
   \\[v \in V, v = (v1, v2, ..., vn) := [vi], vi \in \mathbb { R }, \forall i \in {1 .. n} \\] 
   * To represent data in multi-dimensional space, we have: 
   
   | Rank  | Math Entity                     | Formula  |
   | ----- |:-------------------------------:|:--------:| 
   | 0     | Scalar (magnitude only)         | \\[v \in \mathbb {R} \\]    |
   | 1     | Vector (magnitude and direction)| \\[v \in \mathbb {R}^m, v = [vi] \\]    |
   | 2     | Matrix (table of numbers)       | \\[v \in \mathbb {R}^{ m \times n }, v = [vij] \\]    |
   | 3     | 3-Tensor (cube of numbers)      | \\[v \in \mathbb {R}^{ m \times n \times p }, v = [vijk], vijk \in \mathbb {R}, i \in {1,..,m}, j \in {1,..,n}, k \in {1,..,p} \\]    |
   
   
   4.4. Function space
   \\[f ( z ) = a _ { 0 } + a _ { 1 } z + a _ { 2 } z ^ { 2 } + \cdots + a _ { n } z ^ { n } = \sum _ { i = 0 } ^ { n } a _ { i } z ^ { i } \in P _ { n } ( \mathbb { R } ) \\]
   \\[g ( z ) = b _ { 0 } + b _ { 1 } z + b _ { 2 } z ^ { 2 } + \ldots + b _ { n } z ^ { n } = \sum _ { i = 0 } ^ { n } b _ { i } z ^ { i } \in P _ { n } ( \mathbb { R } ) \\]
   * We suppose f and g are the vectors in vector space \\(P _ { n } ( \mathbb { R } ) \\), so two following operations must be satisfy: 
   \\[h = f + g ; h ( z ) = \sum _ { i = 0 } ^ { n } \left( a _ { i } + b _ { i } \right) z ^ { i } \in P _ { n } ( \mathbb { R } ) \\]
   \\[v = \alpha f ; v ( z ) = \sum _ { i = 0 } ^ { n } \alpha a _ { i } z ^ { \prime } \in P _ { n } ( \mathbb { R } ) \\]
   
     -> \\(P _ { n } ( \mathbb { R } ) \\) is a vector space.
   
   * { \\({ 1 , z , z ^ { 2 } , \ldots , z ^ { n } } \\)} is vector in \\(P _ { n } ( \mathbb { R } ) \\)(all a is 1)
   * \\(\ {\{ e _ { i } \ }\} = \ { z ^ { i } \ } _ { i = 0 } ^ { n } \\) is basis of \\(P _ { n } ( \mathbb { R } ) \\)
   * \\(\forall v \in V = P _ { n } ( \mathbb { R } ) : \quad v = a _ { 0 } e _ { 0 } + \cdots + a _ { n } e _ { n } = \sum _ { i = 0 } ^ { n } a _ { j } e _ { i }\\), v is linear combination of the vector base. 
   * Basis in vector space means:
     * Directions
     * Landmarks
     * Features
     * Words
     * Prototypes, patterns, templates
     * Regularities, abstractions
   * Every vectors in vectors space can be decomposed by the basis vector:  \\(\xi = \left( e _ { 1 } , \dots , e _ { n } \right)\\)(ordered basis)
   \\[\stackrel { \forall v \in V } { \longrightarrow } \text { decomposition } [ v ] \varepsilon = \text { coordinates } \left( a _ { 1 } , \ldots , a _ { n } \right) \in \mathbb { R } ^ { n }\\]
     Each vectors is represented by the basis vectors and coordinate parameters that show the similarity between them (important note).
     
We will continue with basis vectors and coordinate vectors in the next part.
  
I hope you like it!

Source: 

[Machine Learning - Lecture 1](https://www.facebook.com/curiousAI/videos/1431500110327587/?fref=mentions)

[Machine Learning - Lecture 2.1](https://www.facebook.com/curiousAI/videos/1443924629085135/?hc_location=ufi)

[Machine Learning - Lecture 2.2](https://www.facebook.com/curiousAI/videos/1443931999084398/?hc_location=ufi)
 
