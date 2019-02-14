---
layout: post
title: "Data preparation for machine learning"
image: knn-1.png
author: Hai Dang
comments: true
---
# Welcome to my next blog
**Hello**, this is blog about how we prepare the data for machine learning.
### Why and When? 
In five steps for a machine learning problem (T, E, A, P, F), E(Experience) is important and has significant effect on result. In case we input raw data without preprocessing, it can have negative effect on your result. So, prepare data first is really necessary for the problems that have complicated data set. And in this blog, I will present 4 way to prepare data: normalization, standardization, whitening, sphering.

### Mathematical
Data Normalization (feature scaling ):
* \\(x _ { n o r m } = \frac { x - x _ { \min } } { x _ { \max } - x _ { \min } }\\) 
* \\(x ^ { \prime } = \frac { x - \operatorname { mean } ( x ) } { \max ( x ) - \min ( x ) }\\)
* Use case: image data
* Unused case: Long-tail distribution. It is used to describe a case that some part of the distribution is small and often is arbitrary, but in some cases may be specified objectively and have a significant meaning to the model.

Data Standardization:
* \\(x ^ { \prime } = \frac { x - \overline { x } } { \sigma }\\) with \\(\overline { x } = \operatorname { average } ( x )\\) and \\(\sigma\\) is standard deviation.
* Use case: it is used to put all features in the same scale.

PCA Whitening:
* Principle Components Analysis (PCA) is a algorithm to reduce dimensionality to speed up significantly unsupervised feature learning algorithm.
* Input: \\(\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N] \\), \\(x _ i \in \Re ^ { 2 }\\).
* Ouput: 

