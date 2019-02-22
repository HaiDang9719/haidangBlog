---
layout: post
title: "Recommendation System"
image: content-base-recommendation-system-1.jpg
author: Hai Dang
comments: true
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
---
# Welcome to my next blog
**Hello**, this is blog about Content-based Recommendation System.
### Why and When? 
* Long-tail phenomenon: 
 ![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Long_tail.svg/1200px-Long_tail.svg.png)
 *Source: [Wiki](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Long_tail.svg/1200px-Long_tail.svg.png)*
  * Vertical axis: popularity
  * Horizontal axis: items
  * Problems: Physical institutions only provide most popular items to the left of the vertical line while corresponding on-line institutions provide the entire range of items: the tail as well as the popular items -> online-system must be recommend items to individual users instead of showing all available items of the system.
* The key idea of the content-based in recommendation system is that the system will recommend the products that is only related to the topics, articles, categories that the user has rated or interact before. Thus, it is useful in case the users have already rated or show their interest in a product and all products in the system are clearly categorized. 

Content-based method for recommendation system has two negative points: 
* This method waste much information from users and this is bad because users's information is valuable in many cases e.g suggest for people in a group,...
* The content-based method only works well with the tagged items to build item profile.
These drawbacks can be solved by using Neighborhood based collaborative filtering method. 
  * User-user collaborative filtering
  * Item-item collaborative filtering

### Mathematical for content-based recommendation system
* Utility Matrix: matrix with 2 classes of entities, which we shall refer to users and items. We assume that the matrix is sparse, which means most of entries are unknown. 
* How: 
  * Ask users rate the item (can be rate from 1 -> 5)
  * Make inferences from users' behavior (binary value: 1 if users buy or view the item (like) and 0 if vice versa)
* Reprocess data:
  * Build item profile (feature vector for the item): \\(\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N]\\)
* Input: \\(n\\) users, \\(\mathbf{X} = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_m]\\) for \\(m\\) items, utility matrix.
* Task: Find optimal user's model (find value of \\(w\\) and \\(b\\) in linear loss function)
* Output: \\(\theta _ { i }\\) model user \\(i\\)
* Performance: optimal value Loss function
* Algorithm:
\\[\mathcal { L } _ { n } = \frac { 1 } { 2 } \sum _ { m : r _ { m n } = 1 } \left( \mathbf { x } _ { m } \mathbf { w } _ { n } + b _ { n } - y _ { m n } \right) ^ { 2 } + \frac { \lambda } { n } \left\| \mathbf { w } _ { n } \right\| _ { 2 } ^ { 2 }\\]
  
  * \\(\frac { \lambda } { n } \left\| \mathbf { w } _ { n } \right\| _ { 2 } ^ { 2 }\\) is regulation term and \\(\lambda \\) is a positive parameter. 
  * In practical, we can take the average value with \\(s _ { n } = \sum _ { m = 1 } ^ { M } r _ { m n }\\) is the sum of rated items for user \\(n\\).
  
  \\[\mathcal { L } _ { n } = \frac { 1 } { 2 s _ { n } } \sum _ { m : r _ { m n } = 1 } \left( \mathbf { x } _ { m } \mathbf { w } _ { n } + b _ { n } - y _ { m n } \right) ^ { 2 } + \frac { \lambda } { s _ { n } } \left\| \mathbf { w } _ { n } \right\| _ { 2 } ^ { 2 }\\]
  * This formula can be shortened by calculate only on the rated item of user \\(n\\).
    * \\(\hat { \mathbf { x } } _ { n }\\) is sub matrix of feature matrix \\(X\\) (feature matrix of rated items of user \\(n\\))
    * \\(\hat { \mathbf { y } } _ { n }\\) is sub vector of \\(y\\) (only rated value).
  \\[\mathcal { L } _ { n } = \frac { 1 } { 2 s _ { n } } \left\| \hat { \mathbf { X } } _ { n } \mathbf { w } _ { n } + b _ { n } \mathbf { e } _ { n } - \hat { \mathbf { y } } _ { n } \right\| _ { 2 } ^ { 2 } + \frac { \lambda } { 2 s _ { n } } \left\| \mathbf { w } _ { n } \right\| _ { 2 } ^ { 2 }\\]
  
### Mathematical for neighborhood-based collaborative filtering
Similarity function:
* Cosine similarity:
\\[\text { cosine } _ { - } \operatorname { similarity } \left( \mathbf { u } _ { 1 } , \mathbf { u } _ { 2 } \right) = \cos \left( \mathbf { u } _ { 1 } , \mathbf { u } _ { 2 } \right) = \frac { \mathbf { u } _ { 1 } ^ { T } \mathbf { u } _ { 2 } } { \left\| \mathbf { u } _ { 1 } \right\| _ { 2 } \cdot \left\| \mathbf { u } _ { 2 } \right\| _ { 2 } }\\]
* Jaccard: 
\\[J ( u _ 1 , u _ 2 ) = \frac { | u _ 1 \cap u _ 2 | } { | u _ 1 \cup u _ 2 | }\\]

##### Programming step User-User collaborative filtering:

Step 1: Normalized utility matrix 
* Calculate the mean value of each user columns in utility matrix
* Minus each value in user column with the mean value of that column:
  * Positive value means this user like the item
  * Negative value means this user does not like the item
  * Unknown value is represented as 0
  * To optimize the storage, we can store this matrix as a spare matrix that only stores non-zero value and its position.
  
Step 2: Create similarity matrix for users with one of two similarity function above

Step 3: Predict missing value in utility matrix 
* Algorithm: Predict rating of user \\(u\\) for item \\(i\\)
\\[\hat { \boldsymbol { y } } _ { i , u } = \frac { \sum _ { u _ { j } \in \mathcal { N } ( u , i ) } \overline { y } _ { i , u _ { j } } \operatorname { sim } \left( u , u _ { j } \right) } { \sum _ { u _ { j } \in \mathcal { N } ( u , i ) } \left| \operatorname { sim } \left( u , u _ { j } \right) \right| }\\]
  * \\(\mathcal { N } ( u , i )\\) is \\(k\\) users has highest similarity to user \\(u\\) has rated \\(i\\)
  * \\(\overline { \boldsymbol { y } } _ { i , u _ { j } }\\) is rated value in normalised utility matrix of user \\( u _ { j }\\) on item \\(i\\)
  * \\(\operatorname { sim } \left( u , u _ { j } \right)\\) is similarity value between user \\(u\\) and user \\(u _ { j }\\)
  
##### Programming step Item-Item collaborative filtering:

Step 1: Normalized utility matrix 
* Calculate the mean value of each item columns in utility matrix
* Minus each value in item column with the mean value of that column.
* Positive value means this user like the item
  * Negative value means this user does not like the item
  * Unknown value is represented as 0
  * To optimize the storage, we can store this matrix as a spare matrix that only stores non-zero value and its position.
  
Step 2: Create similarity matrix for items with one of two similarity function above

Step 3: Predict missing value in utility matrix 
* Algorithm: Predict rating of user \\(u\\) for item \\(i\\)
\\[\hat { \boldsymbol { y } } _ { i , u } = \frac { \sum _ { i _ { j } \in \mathcal { N } ( i , u ) } \overline { y } _ { i _ { j } , u } \operatorname { sim } \left( i , i _ { j } \right) } { \sum _ { i _ { j } \in \mathcal { N } ( i , u ) } \left| \operatorname { sim } \left( i , i _ { j } \right) \right| }\\]
  * \\(\mathcal { N } ( i , u )\\) is \\(k\\) items has highest similarity to item \\(i\\) has been rated by user \\(u\\)
  * \\(\overline { \boldsymbol { y } } _ { i _ { j } , u }\\) is rated value in normalised utility matrix of item \\( i _ { j }\\) has been rated by user \\(u\\)
  * \\(\operatorname { sim } \left( i , i _ { j } \right)\\) is similarity value between item \\(i\\) and item \\(i _ { j }\\)
  
### Extension

##### Optimize time complexity in computing similarity function for big data

![](../img/recommendation-system-slow-algorithm.jpg)
 *Source: [Giaithuatlaptrinh](http://www.giaithuatlaptrinh.com/?p=1320)*
   
* Problem:
  * We use two loop, so the time complexity in here is: \\(O \left( N ^ { 2 } \right)\\) -> impossible in large number of users (millions of users)
* Task: Optimal time complexity
* Performance: Reduce time complexity to \\(O \left( N \right)\\)
* Algorithm: Using LSH(Locality Sensitive Hashing) function 
* Requirement: if \\(J ( A , B ) \geq s\\), then \\(A, B\\) is belong to a cluster(with high probability)
* Solution: 
![](../img/recommendation-system-fast-1.jpg)
![](../img/recommendation-system-fast-2.jpg)
 *Source: [Giaithuatlaptrinh](http://www.giaithuatlaptrinh.com/?p=1320)*
 
* The MinHash will return the minimum value of permutation index of items viewed by user \\(A\\).
* With the random permutation, the probability that an item has the the minimum index in random permutation \\(S\\) for set \\(M\\) of \\(I\\) items is \\(1 / M\\) -> the probability that \\(h ( A ) = h ( B )\\) is the probability that we select an item (any) in \\(A \cup B\\) and that item has the minimum index value in \\(A \cup B\\).
  \\[\operatorname { Pr } [ h ( A ) = h ( B ) ] = \frac { | A \cap B | } { | A \cup B | }\\] -> this formula is the same is the Jaccard similarity function. 
* Problem of the above solution:
  * The hash function is not satisfy the requirement high probability that two users satisfy the threshold for similarity are belongs to same group. With the current algorithm, for example the threshold is \\(0.5 \\), the probability that two users has similarity value is \\(0.5 \\) is \\(0.5\\).
  * High storage requirement for describing the hash function.
* Solution for the first problem: 
  * Algorithm: 
    * Create \\(K\\) minHash function \\(h _ { 1 } ( . ) , \dots , h _ { K } ( . )\\). Each minHash function uses a different random permutation.
    * Device \\(K\\) into \\(p\\) groups and each group has \\(r\\) functions.
    * The probability that two users \\(A and B\\) belongs to a group is \\(P _ { 1 } = \operatorname { Pr } \left[ h _ { 1 } ( A ) = h _ { 1 } ( B ) \right] \cdot \operatorname { Pr } \left[ h _ { 2 } ( A ) = h _ { 2 } ( B ) \right] \cdots \operatorname { Pr } \left[ h _ { r } ( A ) = h _ { r } ( B ) \right] = x ^ { r }\\)
    * The probability that two users does not belongs to any groups is \\( P _ { 2 } = \left( 1 - x ^ { r } \right) ^ { p } \\).
    -> The probability that two users belongs to at least a group minHash is \\(P _ { 3 } = 1 - \left( 1 - x ^ { r } \right) ^ { p }\\). 
    * Let back to the previous an example with \\( K = 20, p = 10, r = 2, J(A,B) = 0.5 ==> P =  1 - \left( 1 - x ^ { 2 } \right) ^ { 10 } \sim 94 \% \\) is must more larger than the previous probability is \\(0.5\\).
* Solution for the second problem:    
  * Algorithm: Minwise Independent Hashing 

Final solution you can find the sudo-code here:  
![](../img/recommendation-system-fast-3.jpg)
 *Source: [Giaithuatlaptrinh](http://www.giaithuatlaptrinh.com/?p=1320)*
 
##### Matrix Factorization
* Problem: 
  * With content-based recommendation method, we found two drawbacks: this method lose completely users information and it requires a clear item profile matrix. 
  * With the big data, collaborative filtering is not a good solution because it take too much storage for storing the matrix similarity.
* Solution: Matrix Factorization (a kind of combination of two method above):
  * To avoid loosing users information and requirement of known item profile matrix, we let both users model \\(X\\) and item profile \\(W\\) are unknown variable and they are needed to be optimal value. 
    * \\(X\\) is all item profile matrix: \\(\left[ \begin{array} { c } { \mathbf { x } _ { 1 } } \\ { \mathbf { x } _ { 2 } } \\ { \dots } \\ { \mathbf { x } _ { M } } \end{array} \right]\\) and \\(\mathbf { X } \in \mathbb { R } ^ { M \times K }\\).
    * \\(W\\) is all user model matrix: \\(\left[ \begin{array} { l l l l } { \mathbf { w } _ { 1 } } & { \mathbf { w } _ { 2 } } & { \dots } & { \mathbf { w } _ { N } } \end{array} \right]\\) and \\(\mathbf { W } \in \mathbb { R } ^ { K \times N }\\).
    * \\(K\\) represent for latent features of each vector in \\(X, W\\) matrix.
  * The storage need to store \\(X\\) and \\(W\\) is \\(K(M + N)\\) is better than \\(N ^ 2\\) or \\(M ^ 2 \\) in collaborative filtering method. 
  * Function: Quite same as Loss function in content-based method
  \\[\mathcal { L } ( \mathbf { X } , \mathbf { W } ) = \frac { 1 } { 2 s } \sum _ { n = 1 } ^ { N } \sum _ { m : r _ { m m } = 1 } \left( y _ { m n } - \mathbf { x } _ { m } \mathbf { w } _ { n } \right) ^ { 2 } + \frac { \lambda } { 2 } \left( \| \mathbf { X } \| _ { F } ^ { 2 } + \| \mathbf { W } \| _ { F } ^ { 2 } \right)\\]
  * Task: Find \\(\mathbf { X } , \mathbf { W }\\), so that the Loss function has the minimum value.
  * Algorithm: Gradient Descent
  * Detail mathematical solution:
  
  Step1: Fix \\(X\\), Find \\(W\\)
  \\[\mathcal { L } ( \mathbf { W } ) = \frac { 1 } { 2 s } \sum _ { n = 1 } ^ { N } \sum _ { m : r _ { m n } = 1 } \left( y _ { m n } - \mathbf { x } _ { m } \mathbf { w } _ { n } \right) ^ { 2 } + \frac { \lambda } { 2 } \| \mathbf { W } \| _ { F } ^ { 2 }\\]
    * We can see it is a sum of \\(N\\) users, so we can optimize each users first and reduce the form to only rated value with \\(\hat { \mathbf { X } } _ { n }\\) matrix for rated item (row) and \\(\hat { \mathbf { y } } _ { n }\\) rated value.
    \\[\mathcal { L } \left( \mathbf { w } _ { n } \right) = \frac { 1 } { 2 s } \left\| \hat { \mathbf { y } } _ { n } - \hat { \mathbf { X } } _ { n } \mathbf { w } _ { n } \right\| ^ { 2 } + \frac { \lambda } { 2 } \left\| \mathbf { w } _ { n } \right\| _ { 2 } ^ { 2 }\\]
    * Now, we can use derivative method to find \\(w _ {n} \\):
    \\[\mathbf { w } _ { n } = \mathbf { w } _ { n } - \eta \left( - \frac { 1 } { s } \hat { \mathbf { X } } _ { n } ^ { T } \left( \hat { \mathbf { y } } _ { n } - \hat { \mathbf { x } } _ { n } \mathbf { w } _ { n } \right) + \lambda \mathbf { w } _ { n } \right)\\]
  
  Step2: Fix \\(W\\), Find \\(X\\)
  \\[\mathcal { L } ( \mathbf { X } ) = \frac { 1 } { 2 s } \sum _ { n = 1 } ^ { N } \sum _ { m : r _ { m n } = 1 } \left( y _ { m n } - \mathbf { x } _ { m } \mathbf { w } _ { n } \right) ^ { 2 } + \frac { \lambda } { 2 } \| \mathbf { X } \| _ { F } ^ { 2 }\\]
    * We do quite same to step 1, we do the optimization for each vector item (\\(\mathbf { \hat { N } } _ { m }\\) is matrix for rated users (column) and \\(\hat { \mathbf { y } } ^ { m }\\) is rated value)
    \\[\mathcal { L } \left( \mathbf { x } _ { m } \right) = \frac { 1 } { 2 s } \left\| \hat { \mathbf { y } } ^ { m } - \mathbf { x } _ { m } \hat { \mathbf { w } } _ { m } \right\| _ { 2 } ^ { 2 } + \frac { \lambda } { 2 } \left\| \mathbf { x } _ { m } \right\| _ { 2 } ^ { 2 }\\]
    * Now, we can use derivative method to find \\(x _ {m} \\):
    \\[\mathbf { x } _ { m } = \mathbf { x } _ { m } - \eta \left( - \frac { 1 } { s } \left( \hat { \mathbf { y } } ^ { m } - \mathbf { x } _ { m } \mathbf { \hat { W } } _ { m } \right) \mathbf { \hat { w } } _ { m } ^ { T } + \lambda \mathbf { x } _ { m } \right)\\]
    
##### Singular Value Decomposition(SVD)

1. Eigenvalues and eigenvectors: square matrix \\(\mathbf { A } \in \mathbb { R } ^ { n \times n }\\), vector \\(\mathbf { x } \neq \mathbf { 0 } \in \mathbb { R } ^ { n }\\) and a value \\(\lambda\\)
    \\[\mathbf { A } \mathbf { x } = \lambda \mathbf { x }\\]
* If \\(\mathbf { A }\\) is Eigenvectors of \\(\mathbf { x }\\) with eigenvalue \\(\lambda\\) -> \\(k\mathbf { x }\\) is also the the eigenvectors of \\(mathbf { X }\\).
* Square matrix \\(n\\) has \\(n\\) eigenvalues.
* Symmetric matrix, all eigenvalues are the real number.
* Positive definite matrix, all eigenvalues are the positive real number.
* Positive semi-definite matrix, all eigenvalues are the non-negative real number.
2. Orthogonal and orthonormal system
* Orthogonal is \\(\mathbf { u } _ { 1 } , \mathbf { u } _ { 2 } , \ldots , \mathbf { u } _ { m } \in \mathbb { R } ^ { m }\\) and each vector \\(\mathbf { u } _ { i }\\) must satisfy: 
\\[\mathbf { u } _ { i } \neq \mathbf { 0 } ; \quad \mathbf { u } _ { i } ^ { T } \mathbf { u } _ { j } = 0 \forall 1 \leq i \neq j \leq m\\]
* Orthonormal is  \\(\mathbf { u } _ { 1 } , \mathbf { u } _ { 2 } , \ldots , \mathbf { u } _ { m } \in \mathbb { R } ^ { m }\\) and each vector \\(\mathbf { u } _ { i }\\) must satisfy: 
\\[\mathbf { u } _ { i } ^ { T } \mathbf { u } _ { j } =  \begin{Bmatrix}\begin{array} { c c c } { 1 } & { \text { if } } & { i = j } 
\\\\ { 0 } & { \text { otherwise } } \end{array} \end{Bmatrix}\\] 
* Orthogonal matrix: \\(\mathbf { U } = \left[ \mathbf { u } _ { 1 } , \mathbf { u } _ { 2 } , \ldots ,\mathbf { u } _ { m } \right]\\), unitary matrix of \\(m\\) degrees: \\(\mathbf { I }\\)
 \\[\mathbf { U } \mathbf { U } ^ { T } = \mathbf { U } ^ { T } \mathbf { U } = \mathbf { I }\\]
  * \\(\mathbf { U } ^ { - 1 } = \mathbf { U } ^ { T }\\)
  * \\(\mathbf { U }\\) is orthogonal matrix -> \\(\mathbf { U } ^ { T }\\) is also orthogonal matrix.
  * \\(\operatorname { det } ( \mathbf { U } ) = \operatorname { det } \left( \mathbf { U } ^ { T } \right)\\) and \\(\operatorname { det } ( \mathbf { U } ) \operatorname { det } \left( \mathbf { U } ^ { T } \right) = \operatorname { det } ( \mathbf { I } ) = 1\\)
  * Rotate two vectors \\(\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { m }\\) with orthogonal matrix, we get \\(\mathbf { U } \mathbf { x } , \mathbf { U } \mathbf { y }\\) that satisfies: 
  \\[( \mathbf { U } \mathbf { x } ) ^ { T } ( \mathbf { U } \mathbf { y } ) = \mathbf { x } ^ { T } \mathbf { U } ^ { T } \mathbf { U } \mathbf { y } = \mathbf { x } ^ { T } \mathbf { y }\\]
3. Singular Value Decomposition (SVD)

3.1 Definition
\\[\mathbf { A } _ { m \times n } = \mathbf { U } _ { m \times m } \boldsymbol { \Sigma } _ { m \times n } \left( \mathbf { V } _ { n \times n } \right) ^ { T }\\]
* \\(\mathbf { U } , \mathbf { V }\\) is orthogonal matrix
* \\(\boldsymbol { \Sigma }\\) is diagonal trix of non-square matrix \\(\sigma _ { 1 } \geq \sigma _ { 2 } \geq \cdots \geq \sigma _ { r } \geq 0 = 0 = \cdots = 0\\) and \\(r\\) is the rank of the matrix.

3.2 Compact SVD
\\[\mathbf { A } = \sigma _ { 1 } \mathbf { u } _ { 1 } \mathbf { v } _ { 1 } ^ { T } + \sigma _ { 2 } \mathbf { u } _ { 2 } \mathbf { v } _ { 2 } ^ { 2 } + \cdots + \sigma _ { r } \mathbf { u } _ { r } \mathbf { v } _ { r } ^ { T }\\]
* \\(\mathbf { u } _ { 1 } \mathbf { v } _ { i } ^ { T } , 1 \leq i \leq r\\) is matrix with rank is 1.
* 


I hope you like it!

Source: 
 
[Recommendation Systems - standford](http://infolab.stanford.edu/~ullman/mmds/ch9.pdf)

[Content-based Recommendation system](https://machinelearningcoban.com/2017/05/17/contentbasedrecommendersys/)

[Neighborhood-Based Collaborative Filtering](https://machinelearningcoban.com/2017/05/24/collaborativefiltering/)

[Matrix Factorization](https://machinelearningcoban.com/2017/05/31/matrixfactorization/)

[Recommendation system in big data](http://www.giaithuatlaptrinh.com/?p=1320)

[Random Permutaiton](https://en.wikipedia.org/wiki/Random_permutation)

[Singular Value Decomposition](https://machinelearningcoban.com/2017/06/07/svd/)
