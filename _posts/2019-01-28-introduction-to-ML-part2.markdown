---
layout: post
title: "Introduction to AI and Machine Learning - Part2"
image: introduction-to-ml.png
author: Hai Dang
comments: true
---
# Welcome to the introduction series about AI and Machine Learning - Part2
Basis vector
* The key idea of basic vector is to identify a vector in vector space, so it can have different meanings: 
  * Directions
    * Landmarks
    * Features
    * Words
    * Prototypes, patterns, templates
    * Regularities, abstractions
* Each vectors can be decomposed as a linear combination of basic vector \\(\mathcal { E } = \left( e _ { 1 } , \ldots , e _ { n } \right)\\) and parameters(coordinate) with \\( { R } ^ { n }\\) is coordinate space as the diagram below: 
\\[\stackrel { \forall v \in V } { \longrightarrow } \text { decomposition } [ v ] _ { \mathcal { E } } = \text { coordinates } \left( a _ { 1 } , \ldots , a _ { n } \right) \in \mathbb { R } ^ { n }\\] 
* The parameter \\(a _ { i }\\) represents for the similarity between \\(v\\) and \\(e\\), e.g \\(a _ { i } > 1\\) means \\(v _ { i }\\) and \\(e _ { i }\\) are same direction.
* Exampe: \\(\mathbb { R } ^ { n } \supset v = \left( x _ { 1 } , \ldots , x _ { n } \right)\\)
  \\(u _ { 1 } = ( 1,0 , \ldots , 0 ) , \ldots , u _ { n } = ( 0 , \ldots , 0,1 ) \Rightarrow [ v ] u = \left( x _ { 1 } , \ldots , x _ { n } \right)\\)
  * The value \\(u _ { i }\\) is called standard basis vector.
* How about polynomial space: 
\\[P _ { n } ( \mathbb { R } ) \supset v ( z ) = \sum _ { i = 0 } ^ { n } a _ { i } z ^ { i } , a _ { i } \in \mathbb { R } \forall i = 0 , \ldots , n :\\]

\\[\mathcal { E } = \begin{Bmatrix}  e _ { i } = z ^ { i } \end{Bmatrix} _ { i = 0 } ^ { n } \Rightarrow [ v ] \varepsilon = \left( a _ { 0 } , \ldots , a _ { n } \right) \in \mathbb { R } ^ { n + 1 }\\]

  * \\(V \stackrel { \mathcal { E } } { \rightarrow } \mathbb { R } ^ { n }\\): Every vector space through basis vectors \\(\mathcal { E }\\) can be transformed to vectors in \\(\mathbb { R } ^ { n }\\).
  * Dimension \\(n\\) is the minimum number of basis vectors to represent \\(\forall v \in V\\).
Linear transformation <-> matrix
![]()
   
       



I hope you like it!

Source: 
