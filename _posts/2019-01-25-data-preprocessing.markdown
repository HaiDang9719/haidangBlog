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

Data Standardization:
* \\(x ^ { \prime } = \frac { x - \overline { x } } { \sigma }\\) with \\(\overline { x } = \operatorname { average } ( x )\\) and \\(\sigma\\) is standard deviation.

