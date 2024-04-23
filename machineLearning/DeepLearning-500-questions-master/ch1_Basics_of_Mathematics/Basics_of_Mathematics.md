\[TOC\]

## Chapter 1 Mathematical Basics

What mathematical foundations are usually required for deep learning? What is so difficult about mathematics in deep learning? Beginners usually have these questions. In online recommendations and book recommendations, we often see a series of mathematical subjects listed, such as calculus, linear algebra, probability theory, complex variable functions, numerical calculations, optimization theory, information theory, etc. . These mathematical knowledge are relevant, but in fact, learning according to this scope of knowledge will take a long time and be very boring. In this chapter, we select some concepts that are easily confused in the basics of mathematics to introduce them to help everyone better understand them. Clarify the relationship between these confusing concepts.

## 1.1 Vectors and matrices

### 1.1.1 The connection between scalars, vectors, matrices and tensors

**scalar**
A scalar represents a single number, unlike most other objects studied in linear algebra (usually arrays of numbers). We represent scalars in italics. Scalars are usually given lowercase variable names.

**vector**
A vector represents an ordered set of numbers. Through the index in the sequence, we can determine each individual number. Usually we give vectors bold lowercase variable names, such as xx. Elements in a vector can be represented by italics with subscripts. The first element of vector $X$ is $X\_1$, the second element is $X\_2$, and so on. We also note the type of elements stored in the vector (real, imaginary, etc.).

**matrix**
A matrix is a collection of objects with the same characteristics and dimensions, represented as a two-dimensional data table. The meaning is that an object is represented as a row in the matrix, a feature is represented as a column in the matrix, and each feature has a numerical value. Matrices are usually given bold uppercase variable names, such as $A$.

**Tensor**
In some cases, we will discuss arrays with coordinates in more than two dimensions. Generally, the elements in an array are distributed in a regular grid of several dimensional coordinates, which we call a tensor. use $A$to represent tensor "A". The element with coordinates $(i,j,k)$ in tensor $A$ is denoted as $A\_{(i,j,k)}$.

**The relationship between the four**

> A scalar is a tensor of order 0, and a vector is a tensor of order 1. Example:
> Scalar means you know the length of the stick, but you don't know where the stick is pointing.
> The vector not only knows the length of the stick, but also knows whether the stick is pointing forward or backward.
> The tensor not only knows the length of the stick, but also knows whether the stick is pointing forward or backward, and how much the stick is deflected up/down and left/right.

### 1.1.2 The difference between tensors and matrices

-   Algebraically speaking, matrices are the generalization of vectors. A vector can be regarded as a one-dimensional "table" (that is, the components are arranged in a row in order), and the matrix is a two-dimensional "table" (the components are arranged according to their vertical and horizontal positions), then the $n$-order tensor is the so-called $n$-dimensional "form". The strict definition of a tensor is to use a linear map to describe it.
-   From a geometric point of view, the matrix is a real geometric quantity, that is, it is something that does not change with the coordinate transformation of the reference system. Vectors also have this property.
-   Tensors can be expressed in the form of a 3×3 matrix.
-   Numbers representing scalars and three-dimensional arrays representing vectors can also be viewed as 1×1 and 1×3 matrices respectively.

### 1.1.3 Matrix and vector multiplication results

If the Einstein summation convention is used, the matrix $A$, $B$The matrix $C$ obtained by multiplication can be expressed by the following formula: $$ a\_{ik}\*b\_{kj}=c\_{ij} \\tag{1.3-1} $$ Among them, $a\_{ik}$, $b_{kj}$, $c_{ij}$Represents the elements of matrices $A, B, and C$ respectively. $k$ appears twice and is a dummy variable (Dummy Variables) indicating the traversal and summation of the parameters. The multiplication of matrices and vectors can be regarded as a special case of matrix multiplication. For example: matrix $B$ is a matrix of $n \\times 1$.

### 1.1.4 Norm induction of vectors and matrices

**The norm of the vector (norm)**
Define a vector as: $\\vec{a}=\[-5, 6, 8, -10\]$. Any set of vectors is set to $\\vec{x}=(x\_1,x\_2,...,x\_N)$. Its different norms are solved as follows:

-   The 1 norm of a vector: the sum of the absolute values of each element of the vector. The result of the 1 norm of the above vector $\\vec{a}$ is: 29.

$$
\Vert\vec{x}\Vert_1=\sum_{i=1}^N\vert{x_i}\vert
$$

-   2-norm of the vector: Taking the sum of the squares of each element of the vector and then taking the square root, the result of the 2-norm of the above $\\vec{a}$ is: 15.

$$
\Vert\vec{x}\Vert_2=\sqrt{\sum_{i=1}^N{\vert{x_i}\vert}^2}
$$

-   The negative infinity norm of a vector: the smallest absolute value of all elements of the vector: the result of the negative infinity norm of the above vector $\\vec{a}$ is: 5.

$$
\Vert\vec{x}\Vert_{-\infty}=\min{|{x_i}|}
$$

-   The positive infinity norm of a vector: the largest absolute value of all elements of the vector: the result of the positive infinity norm of the above vector $\\vec{a}$ is: 10.

$$
\Vert\vec{x}\Vert_{+\infty}=\max{|{x_i}|}
$$

-   p-norm of a vector:

$$
L_p=\Vert\vec{x}\Vert_p=\sqrt[p]{\sum_{i=1}^{N}|{x_i}|^p}
$$

**norm of matrix**

Define a matrix $A=\[-1, 2, -3; 4, -6, 6\]$. Any matrix is defined as: $A\_{m\\times n}$, whose elements are $a_{ij}$。

The norm of a matrix is defined as

$$
\Vert{A}\Vert_p :=\sup_{x\neq 0}\frac{\Vert{Ax}\Vert_p}{\Vert{x}\Vert_p}
$$

When vectors take different norms, different matrix norms are obtained accordingly.

-   **1 norm of the matrix (column norm)** : the elements in each column of the matrix

    The prime absolute values are summed first, and then the largest one is taken (the column sum is the largest). The 1 norm of the above matrix $A$ is first obtained $\[5,8,9\]$, and then the maximum final result is: 9 . $$ \\Vert A\\Vert\_1=\\max\_{1\\le j\\le n}\\sum\_{i=1}^m|{a\_{ij}}| $$

-   **2-norm of the matrix** : Taking the square root of the maximum eigenvalue of the matrix $A^TA$, the final result obtained by the 2-norm of the above matrix $A$ is: 10.0623.


$$
\Vert A\Vert_2=\sqrt{\lambda_{max}(A^T A)}
$$

in, $\lambda_{max}(A^T A)$ for $A^T A$The maximum absolute value of the eigenvalue.

-   **Infinite norm of the matrix (row norm)** : The absolute values of the elements in each row of the matrix are first summed, and then the largest one is taken (the row sum is the largest). The row norm of the above matrix $A$ is first obtained $\[ 6;16\]$, and then take the maximum final result: 16. $$ \\Vert A\\Vert\_{\\infty}=\\max\_{1\\le i \\le m}\\sum\_{j=1}^n |{a\_{ij}}| $$

-   **The nuclear norm of the matrix** : the sum of the singular values of the matrix (decomposing the matrix svd). This norm can be used for low-rank representation (because minimizing the nuclear norm is equivalent to minimizing the rank of the matrix - low rank), as mentioned above The final result of matrix A is: 10.9287.

-   **L0 norm of the matrix** : the number of non-zero elements of the matrix. It is usually used to indicate sparseness. The smaller the L0 norm, the more zero elements there are and the sparser it is. The final result of the above matrix $A$ is: 6.

-   **The L1 norm of the matrix** : the sum of the absolute values of each element in the matrix. It is the optimal convex approximation of the L0 norm, so it can also represent sparseness. The final result of the above matrix $A$ is: 22.

-   **The F norm of the matrix** : the sum of the squares of each element of the matrix and then the square root. It is usually also called the L2 norm of the matrix. Its advantage is that it is a convex function that can be derived and solved, and is easy to calculate. The final result of the above matrix A That is: 10.0995.


$$
\Vert A\Vert_F=\sqrt{(\sum_{i=1}^m\sum_{j=1}^n{| a_{ij}|}^2)}
$$

-   **L21 norm of the matrix** : The matrix first calculates the F norm of each column in units of each column (which can also be considered as the 2 norm of the vector), and then calculates the L1 norm of the obtained result (which can also be considered as the vector's 2 norm). 1 norm), it is easy to see that it is a norm between L1 and L2. The final result of the above matrix $A$ is: 17.1559.
-   **p norm of matrix**

$$
\Vert A\Vert_p=\sqrt[p]{(\sum_{i=1}^m\sum_{j=1}^n{| a_{ij}|}^p)}
$$

### 1.1.5 How to determine whether a matrix is positive definite

To determine whether a matrix is positive definite, there are usually the following aspects:

-   The main sub-expressions of the sequence are all greater than 0;
-   There is an invertible matrix $C$ such that $C^TC$ is equal to this matrix;
-   The positive inertia index is equal to $n$;
-   Contracted with the identity matrix $E$ (that is: the canonical form is $E$)
-   The principal and diagonal elements in the standard form are all positive;
-   The eigenvalues are all positive;
-   is the metric matrix of a certain basis.

## 1.2 Derivatives and partial derivatives

### 1.2.1 Calculation of partial derivatives

**Derivative definition** :

The derivative represents the ratio of the change in the function value to the change in the independent variable when the change in the independent variable tends to be infinitesimal. The geometric meaning is the tangent to this point. The physical meaning is the (instantaneous) rate of change at that moment.

_Note_ : In a one-variable function, only one independent variable changes, which means there is only a rate of change in one direction. This is why a one-variable function has no partial derivatives. In physics, there are mean speed and instantaneous speed. The average speed is

$$
v=\frac{s}{t}
$$

Where $v$ represents the average speed, $s$ represents the distance, and $t$ represents the time. This formula can be rewritten as

$$
\bar{v}=\frac{\Delta s}{\Delta t}=\frac{s(t_0+\Delta t)-s(t_0)}{\Delta t}
$$

Where $\\Delta s$ represents the distance between two points, and $\\Delta t$ represents the time it takes to travel this distance. When $\\Delta t$ tends to 0 ($\\Delta t \\to 0$), that is, when time becomes very short, the average speed becomes the instantaneous speed at time $t\_0$, expressed in the following form :

$$
v(t_0)=\lim_{\Delta t \to 0}{\bar{v}}=\lim_{\Delta t \to 0}{\frac{\Delta s}{\Delta t}}=\lim_{\Delta t \to 0}{\frac{s(t_0+\Delta t)-s(t_0)}{\Delta t}}
$$

In fact, the above formula represents the derivative of the function of distance $s$ with respect to time $t$ at $t=t\_0$. Generally, the derivative is defined like this: If the limit of the average rate of change exists, that is,

$$
\lim_{\Delta x \to 0}{\frac{\Delta y}{\Delta x}}=\lim_{\Delta x \to 0}{\frac{f(x_0+\Delta x)-f(x_0)}{\Delta x}}
$$

Then this limit is called a function $y=f(x)$at point $x_0$derivative at . Referred to as $f'(x_0)$ or $y'\vert_{x=x_0}$ or $\frac{dy}{dx}\vert_{x=x_0}$ or $\frac{df(x)}{dx}\vert_{x=x_0}$。

In layman's terms, the derivative is the slope of the tangent line to the curve at a certain point.

**Partial derivative** :

Since we talk about partial derivatives, at least two independent variables are involved. Take two independent variables as an example, $z=f(x,y) $, from derivatives to partial derivatives, that is, from curves to surfaces. A point on a curve has only one tangent line. But at a point on a curved surface, there are countless tangent lines. The partial derivative refers to the rate of change of a multivariate function along the coordinate axis.

_Note_ : Intuitively, the partial derivative is the rate of change of the function along the positive direction of the coordinate axis at a certain point.

Assume that function $z=f(x,y) $ is defined in the domain of point $(x\_0,y\_0) $. When $y=y\_0 $, $z $ can be regarded as a one-variable function $f about $x $. (x,y\_0) $, if the one-variable function is differentiable at $x=x\_0 $, then we have

$$
\lim_{\Delta x \to 0}{\frac{f(x_0+\Delta x,y_0)-f(x_0,y_0)}{\Delta x}}=A
$$

The limit $A$ of the function exists. Then $A$ is called the partial derivative of function $z=f(x,y)$ at point $(x\_0,y\_0)$ with respect to independent variable $x$, which is denoted as $f\_x(x\_0,y\_0)$ or $\\ frac{\\partial z}{\\partial x}\\vert\_{y=y\_0}^{x=x\_0}$ or $\\frac{\\partial f}{\\partial x}\\vert\_{y=y\_0}^{x= x\_0}$ or $z\_x\\vert\_{y=y\_0}^{x=x\_0}$.

When solving partial derivatives, you can regard another variable as a constant and use ordinary derivative methods to solve it. For example, the partial derivative of $z=3x^2+xy$ with respect to $x$ is $z\_x=6x+y$. This When $y$ is equivalent to the coefficient of $x$.

The geometric meaning of the partial derivative at a certain point $(x\_0,y\_0)$ is that the intersection line of surface $z=f(x,y)$ and surface $x=x\_0$ or surface $y=y\_0$ is at $y=y\_0$ Or the slope of the tangent line at $x=x\_0$.

### 1.2.2 What is the difference between derivatives and partial derivatives?

There is no essential difference between derivatives and partial derivatives. If a limit exists, it is the limit of the ratio of the change in the function value to the change in the independent variable when the change in the independent variable approaches 0.

> -   For a one-variable function, one $y$ corresponds to one $x$, and there is only one derivative.
> -   A binary function, a $z$ corresponds to a $x$ and a $y$, has two derivatives: one is the derivative of $z$ with respect to $x$, and the other is the derivative of $z$ with respect to $y$, which is called is a deflection.
> -   When seeking partial derivatives, it should be noted that when seeking the derivative of one variable, the other variable is regarded as a constant and only the change is differentiated, thereby converting the solution of the partial derivative into the derivation of a one-variable function.

## 1.3 Eigenvalues and eigenvectors

### 1.3.1 Eigenvalue decomposition and eigenvectors

-   Eigenvalue decomposition can obtain eigenvalues and eigenvectors;

-   The eigenvalue represents how important the feature is, and the eigenvector represents what the feature is.

    If a vector $\\vec{v}$ is the eigenvector of the square matrix $A$, it must be expressed in the following form:


$$
A\nu = \lambda \nu
$$

$\lambda$is the eigenvalue corresponding to the eigenvector $\\vec{v}$. Eigenvalue decomposition is to decompose a matrix into the following form:

$$
A=Q\sum Q^{-1}
$$

Among them, $Q$ is a matrix composed of eigenvectors of this matrix $A$, $\\sum$ is a diagonal matrix, each diagonal element is an eigenvalue, and the eigenvalues inside are arranged from large to small. , the eigenvectors corresponding to these eigenvalues describe the change direction of this matrix (from major changes to minor changes). That is to say, the information of matrix $A$ can be represented by its eigenvalues and eigenvectors.

### 1.3.2 What is the relationship between singular values and eigenvalues?

So how do singular values and eigenvalues correspond? We multiply the transpose of a matrix $A$ by $A$ and find the eigenvalues of $A^TA$, which has the following form:

$$
(A^TA)V = \lambda V
$$

Here $V$ is the right singular vector above, and there are:

$$
\sigma_i = \sqrt{\lambda_i}, u_i=\frac{1}{\sigma_i}AV
$$

$\\sigma $ here is the singular value, and $u $ is the left singular vector mentioned above. \[The guy who proved it didn’t give it either\] The singular values $\\sigma $ are similar to the eigenvalues. They are also arranged from large to small in the matrix $\\sum $, and the reduction of $\\sigma $ is particularly fast. In many cases, the top 10% or even 1% of the singular values The sum accounts for more than 99% of the sum of all singular values. In other words, we can also use the first $r $ ($r $ is much smaller than $m, n $) singular values to approximately describe the matrix, that is, partial singular value decomposition: $$ A\_{m\\times n}\\approx U\_{m \\times r}\\sum\_{r\\times r}V\_{r \\times n}^T $$

The result of the multiplication of the three matrices on the right will be a matrix close to $A$. Here, the closer $r$ is to $n$, the closer the multiplication result is to $A$.

## 1.4 Probability distribution and random variables

### 1.4.1 Why machine learning uses probability

The probability of an event is a measure of the likelihood of that event occurring. Although the occurrence of an event in a random experiment is accidental, random experiments that can be repeated in large numbers under the same conditions often show obvious quantitative patterns.
In addition to dealing with uncertain quantities, machine learning also needs to deal with random quantities. Uncertainty and randomness can come from many sources, and probability theory is used to quantify uncertainty.
Probability theory plays a central role in machine learning because the design of machine learning algorithms often relies on probabilistic assumptions about the data.

> For example, in the machine learning (Andrew Ng) class, there will be a naive Bayes hypothesis which is an example of conditional independence. The learning algorithm makes assumptions about the content and is used to tell whether an email is spam. Suppose that the probability condition of word x appearing in an email is independent of word y, regardless of whether the email is spam or not. It is clear that this assumption is not without loss of generality, since certain words almost always occur at the same time. However, the end result is that this simple assumption has little impact on the results and allows us to quickly identify spam anyway.

### 1.4.2 What is the difference between variables and random variables?

**random** variable

A real-valued function (all possible sample points) that represents various results in random phenomena (under certain conditions, the phenomenon that does not always have the same result is called a random phenomenon). For example, the number of passengers waiting at a bus stop within a certain period of time, the number of calls received by a telephone exchange within a certain period of time, etc. are all examples of random variables.
The essential difference between the uncertainty of random variables and fuzzy variables is that the measurement results of the latter are still uncertain, that is, fuzziness.

**The difference between variables and random variables:**
When the probability of a variable's value is not 1, the variable becomes a random variable; when the probability of the random variable's value is 1, the random variable becomes a variable.

> for example:
> When the probability that variable $x$ is 100 is 1, then $x=100$ is determined and will not change unless further operations are performed. When the probability of variable $x$ being 100 is not 1, for example, the probability of 50 is 0.5, and the probability of 100 is 0.5, then this variable will change with different conditions and is a random variable. If it is 50 or The probability of 100 is 0.5, which is 50%.

### 1.4.3 The connection between random variables and probability distributions

A random variable only represents a possible state, and the probability of each state must be formulated given the accompanying probability distribution. The method used to describe the likelihood of each possible state of a random variable or a cluster of random variables is **probability distribution** .

Random variables can be divided into discrete random variables and continuous random variables.

The corresponding function describing its probability distribution is

Probability Mass Function (PMF): describes the probability distribution of discrete random variables, usually with capital letters $P$express.

Probability Density Function (PDF): describes the probability distribution of a continuous random variable, usually represented by the lowercase letter $p$.

### 1.4.4 Discrete random variables and probability mass functions

PMF maps each state that a random variable can take to the probability that the random variable takes that state.

-   In general, $P(x) $ represents the probability when $X=x $.
-   Sometimes in order to prevent confusion, it is necessary to clearly write the name of the random variable $P( $x$=x) $
-   Sometimes it is necessary to define a random variable first, and then formulate the probability distribution x it follows to obey $P( $x $) $

PMF can act on multiple random variables at the same time, that is, joint probability distribution (joint probability distribution) $P(X=x,Y=y)$\*express $X=x$The probability of happening simultaneously with $Y=y$ can also be abbreviated as $P(x,y)$.

If a function $P$ is a random variable $X$PMF, then it must satisfy the following three conditions

-   $P$The domain of must be the set of all possible states of x
-   $∀x∈ $x, $0 \leq P(x) \leq 1 $.
-   $∑_{x∈X} P(x)=1$. We call this property normalized.

### 1.4.5 Continuous random variables and probability density functions

If a function $p$ is the PDF of x, then it must satisfy the following conditions

-   $p$The domain of must be the set of all possible states of x.
-   $∀x∈X,p(x)≥0$. Note that we do not require $p(x)≤1$, because here $p(x)$It does not represent the specific probability corresponding to this state, but a relative size (density) of the probability. The specific probability needs to be calculated by integration.
-   $∫p(x)dx=1$, after integration, the sum is still 1, and the sum of probabilities is still 1.

Note: PDF$p(x)$ does not directly give the probability of a specific state, but gives the density. On the contrary, it gives the probability that the area falls on $δx$The probability in the wirelessly small area is $ p(x)δx$. From this, we cannot find the probability of a specific state. What we can find is a certain state. $x$The probability of falling within a certain interval $\[a,b\]$ is $ \\int\_{a}^{b}p(x)dx$.

### 1.4.6 Use examples to understand conditional probability

The conditional probability formula is as follows: $$ P(A|B) = P(A\\cap B) / P(B) $$ Description: For events or subsets $A$ and $B$ in the same sample space $\\Omega$, if an element randomly selected from $\\Omega$ belongs to $B$, then the next randomly selected element The probability of belonging to $A$ is defined as the conditional probability of $A$ under the premise of $B$. The conditional probability Venn diagram is shown in Figure 1.1.
[![条件概率](img/ch1/conditional_probability.jpg)](https://github.com/CodeWithArtemis/Resource-hub/blob/master/machineLearning/DeepLearning-500-questions-master/ch01_%E6%95%B0%E5%AD%A6%E5%9F%BA%E7%A1%80/img/ch1/conditional_probability.jpg)

Figure 1.1 Schematic diagram of conditional probability Venn diagram

According to the Venn diagram, it can be clearly seen that when event B occurs, the probability of event A occurring is $P(A\\bigcap B)$ divided by $P(B)$.
For example: A couple has two children. If one of them is a girl, what is the probability that the other one is a girl? (Encountered both interviews and written tests)
**Exhaustive method** : It is known that one of them is a girl, then the sample space is male and female, female female, female male, then the probability that the other one is still a girl is 1/3.
**Conditional probability method** : $P(female|female)=P(female)/P(female)$. If a couple has two children, then its sample space is female-female, male-female, female-male, male-male, then $P (Female)$ is 1/4, $P (Female) = 1-P(Male)=3/4$, so the final $1/3$.
You may misunderstand here that men and women and women and men are the same situation, but in fact, siblings and brothers and sisters are different situations.

### 1.4.7 The difference between joint probability and marginal probability

**the difference:**
Joint probability: Joint probability refers to the probability similar to $P(X=a,Y=b)$, which contains multiple conditions and all conditions are true at the same time. Joint probability refers to the probability that multiple random variables meet their respective conditions in a multivariate probability distribution.
Marginal probability: Marginal probability is the probability of an event occurring regardless of other events. Marginal probability refers to he probability similar to $P(X=a)$, $P(Y=b)$, which is only related to a single random variable.

**connect:**
The joint distribution can be used to find the marginal distribution, but if only the marginal distribution is known, the joint distribution cannot be found.

### 1.4.8 Chain Rule of Conditional Probability

From the definition of conditional probability, the following multiplication formula can be directly derived:
Multiplication formula Assume $A, B$ are two events, and $P(A) > 0$, then there is $$ P(AB) = P(B|A)P(A) $$ Promote$$ P(ABC)=P(C|AB)P(B|A)P(A) $$ Generally, it can be proved by induction: if $P(A\_1A\_2...A\_n)>0$, then there is $$ P(A\_1A\_2...A\_n)=P(A\_n|A\_1A\_2...A\_{n-1})P(A\_{n-1}|A\_1A\_2...A\_{n-2})...P(A\_2 |A\_1)P(A\_1) =P(A\_1)\\prod\_{i=2}^{n}P(A\_i|A\_1A\_2...A\_{i-1}) $$ Any joint probability distribution of multidimensional random variables can be decomposed into the form of conditional probability multiplication of only one variable.

### 1.4.9 Independence and conditional independence

**Independence:** two random variables $x$ and $y$, the probability distribution is expressed in the form of a product of two factors, one factor only contains $x$, the other factor only contains $y$, the two random variables are independent of each other (independent) .
Conditions sometimes bring independence between events that are not independent, and sometimes they cause originally independent events to lose their independence because of the existence of this condition.
Example: $P(XY)=P(X)P(Y)$, event $X$ and event $Y$ are independent. At this time, given $Z$, $$ P(X,Y|Z) \\not = P(X|Z)P(Y|Z) $$ When events are independent, the joint probability is equal to the product of probabilities. This is a very good mathematical property, but unfortunately, unconditional independence is very rare, because in most cases, events affect each other.

**conditional independence**
Given $Z$, $X$ and $Y$ are conditionally independent if and only if $$ X\\bot Y|Z \\iff P(X,Y|Z) = P(X|Z)P(Y|Z) $$ $X$The relationship with $Y$ depends on $Z$, rather than arising directly.

> **For example,** define the following events:
> $X$: It will rain tomorrow;
> $Y$: The ground is wet today;
> $Z$: Whether it rains today;
> $Z$The establishment of the event has an impact on both $X$ and $Y$. However, under the premise of the establishment of the $Z$ event, today's ground conditions have no impact on whether it will rain tomorrow.

## 1.5 Common probability distributions

### 1.5.1 Bernoulli distribution

**Bernoulli distribution** (Bernoulli distribution, 0-1 distribution) is a single binary random variable distribution, controlled by a single parameter $\\phi$∈\[0,1\], $\\phi$ gives the probability that the random variable is equal to 1. Main properties have: $$ \\begin{align\*} P(x=1) &= \\phi \\ P(x=0) &= 1-\\phi \\ Probability mass function: P(x=x) &= \\phi^x(1-\\phi)^{1-x} \\ \\end{align\*} $$ Its expectation and variance are: $$ \\begin{align\*} E\_x\[x\] &= \\phi \\ Var\_x(x) &= \\phi{(1-\\phi)} \\end{align\*} $$ **Scope of application** : **Bernoulli distribution** is suitable for **modeling discrete** random variables.

**Multinoulli distribution** , also called **category distribution** , is a random distribution of a single _k_ value, often used to represent **the distribution of object classification** . Where $k$ is a finite value. The Multinoulli distribution is represented by the vector $\\vec{p}\\in\[0,1\]^{ k-1}$ parameterization, each component $p\_i$ represents the probability of the $i$-th state, and $p\_k=1-1^Tp$. Here $1^T$ represents the transformation of a column vector whose elements are all 1 The position is actually the sum of the probabilities except k in the vector p. It can be rewritten as $p\_k=1-\\sum\_{0}^{k-1}p\_i$.

Supplement binomial distribution and multinomial distribution:

Binomial distribution, generally a coin is tossed many times. Binomial distribution is **n-fold Bernoulli trials .** the discrete probability distribution of the number of successful

Multinomial Distribution is a generalization of the binomial distribution. Binomial is done n times Bernoulli experiments, which stipulates that there are only two results for each trial. If we still do n times experiments, there can be m results for each trial, and the probability of m results occurring Mutually exclusive and the sum is 1, the probability of one of the results occurring X times is a polynomial distribution.

### 1.5.2 Gaussian distribution

Gaussian is also called Normal Distribution. The probability function is as follows:
$$ N(x;\\mu,\\sigma^2) = \\sqrt{\\frac{1}{2\\pi\\sigma^2}}exp\\left ( -\\frac{1}{2\\sigma^2}(x-\\mu)^2 \\right ) $$ 其中, $\mu$and $\\sigma $ are the mean and standard deviation respectively. The x-coordinate of the central peak is given by $\\mu $. The width of the peak is controlled by $\\sigma $. The maximum point is obtained at $x=\\mu $, and the inflection point is $x. =\\mu\\pm\\sigma $

In the normal distribution, the probabilities under ±1$\\sigma$, ±2$\\sigma$, and ±3$\\sigma$ are 68.3%, 95.5%, and 99.73% respectively. It is best to remember these three numbers.

In addition, let $\\mu=0,\\sigma=1 $ Gaussian distribution is simplified to standard normal distribution: $$ N(x;\\mu,\\sigma^2) = \\sqrt{\\frac{1}{2\\pi}}exp\\left ( -\\frac{1}{2}x^2 \\right ) $$ Efficiently evaluate probability density functions: $$ N(x;\\mu,\\beta^{-1})=\\sqrt{\\frac{\\beta}{2\\pi}}exp\\left(-\\frac{1}{2}\\beta(x-\\ mu)^2\\right) $$

Among them, $\\beta=\\frac{1}{\\sigma^2}$ controls the distribution accuracy through the parameter $\\beta∈(0,\\infty)$.

### 1.5.3 When to use the normal distribution

Q: When is the normal distribution used? Answer: When you lack prior knowledge of the distribution of real numbers and don’t know which form to choose, it is always correct to choose the normal distribution by default. The reasons are as follows:

1.  The central limit theorem tells us that many independent random variables approximately obey a normal distribution. In reality, many complex systems can be modeled as normally distributed noise, even if the system can be structurally decomposed.
2.  The normal distribution is the distribution with the greatest uncertainty among all probability distributions with the same variance. In other words, the normal distribution is the distribution with the least prior knowledge added to the model.

Generalization of normal distribution: The normal distribution can be extended to $R^n$ space, which is called **multi-digit normal distribution** , and its parameter is a positive definite symmetric matrix $\\Sigma $: $$ N(x;\\vec\\mu,\\Sigma)=\\sqrt{\\frac{1}{(2\\pi)^ndet(\\Sigma)}}exp\\left(-\\frac{1}{2}(\\ vec{x}-\\vec{\\mu})^T\\Sigma^{-1}(\\vec{x}-\\vec{\\mu})\\right) $$ Efficient evaluation of probability densities for mostly normal distributions: $$ N(x;\\vec{\\mu},\\vec\\beta^{-1}) = \\sqrt{det(\\vec\\beta)}{(2\\pi)^n}exp\\left(-\\frac{ 1}{2}(\\vec{x}-\\vec\\mu)^T\\beta(\\vec{x}-\\vec\\mu)\\right) $$ Here, $\\vec\\beta$ is a precision matrix.

### 1.5.4 Exponential distribution

In deep learning, exponential distribution is used to describe the distribution of boundary points obtained at $x=0$ point. The exponential distribution is defined as follows: $$ p(x;\\lambda)=\\lambda I\_{x\\geq 0}exp(-\\lambda{x}) $$ The exponential distribution uses the indicator function $I\_{x\\geq 0} $ to make the probability of $x $ taking a negative value zero.

### 1.5.5 Laplace distribution (Laplace distribution)

A closely related probability distribution is the Laplace distribution, which allows us to $\mu$Set the peak value of probability mass at $$ Laplace(x;\\mu;\\gamma)=\\frac{1}{2\\gamma}exp\\left(-\\frac{|x-\\mu|}{\\gamma}\\right) $$

### 1.5.6 Dirac distribution and empirical distribution

The Dirac distribution ensures that all mass in the probability distribution is concentrated at one point. The Dirac delta function of the Dirac distribution (also known as **the unit impulse function** ) is defined as follows: $$ p(x)=\\delta(x-\\mu), x\\neq \\mu $$

$$
\int_{a}^{b}\delta(x-\mu)dx = 1, a < \mu < b
$$

The Dirac distribution often appears as a component of the empirical distribution$$ \\hat{p}(\\vec{x})=\\frac{1}{m}\\sum\_{i=1}^{m}\\delta(\\vec{x}-{\\vec{x}}^{ (i)}) $$ , where m points $x^{1},...,x^{m}$ are a given data set, and **the empirical distribution** assigns the probability density $\\frac{1}{m} $ to these points .

When we train the model on the training set, we can think of the empirical distribution obtained from this training set as indicating **the sampling source** .

**Scope of application** : The Dirac delta function is suitable for **the empirical distribution of continuous** random variables.

## 1.6 Expectation, variance, covariance, correlation coefficient

### 1.6.1 Expectations

In probability theory and statistics, the mathematical expectation (or mean, also simply expectation) is the sum of the probabilities of each possible outcome in an experiment multiplied by its outcome. It reflects the average value of the random variable.

-   Linear operations: $E(ax+by+c) = aE(x)+bE(y)+c$
-   Promotion form: $E(\sum_{k=1}^{n}{a_ix_i+c}) = \sum_{k=1}^{n}{a_iE(x_i)+c}$
-   Function expectation: Suppose $f(x)$ is a function of $x$, then the expectation of $f(x)$ is
    -   Discrete function: $E(f(x))=\sum_{k=1}^{n}{f(x_k)P(x_k)}$
    -   Continuous function: $E(f(x))=\int_{-\infty}^{+\infty}{f(x)p(x)dx}$

> Notice:
>
> -   The expectation of the function is greater than or equal to the expected function (Jensen's inequality, that is, $E(f(x))\\geqslant f(E(x))$
> -   In general, the expected product is not equal to the expected product.
> -   If $X$ and $Y$ are independent of each other, then $E(xy)=E(x)E(y) $.

### 1.6.2 Variance

In probability theory, variance is used to measure the deviation between a random variable and its mathematical expectation (ie, mean). Variance is a special kind of expectation. defined as:

$$
Var(x) = E((x-E(x))^2)
$$

> Variance properties:
>
> 1）$Var(x) = E(x^2) -E(x)^2$
> 2) The variance of the constant is 0;
> 3) The variance does not satisfy linear properties;
> 4) If $X$ and $Y$ are independent of each other, $Var(ax+by)=a^2Var(x)+b^2Var(y)$

### 1.6.3 Covariance

Covariance is a measure of the strength of linear correlation between two variables and the scale of the variables. The covariance of two random variables is defined as: $$ Cov(x,y)=E((xE(x))(yE(y))) $$

Variance is a special kind of covariance. When $X=Y$, $Cov(x,y)=Var(x)=Var(y)$.

> Covariance properties:
>
> 1) The covariance of the independent variable is 0.
> 2) Covariance calculation formula:

$$
Cov(\sum_{i=1}^{m}{a_ix_i}, \sum_{j=1}^{m}{b_jy_j}) = \sum_{i=1}^{m} \sum_{j=1}^{m}{a_ib_jCov(x_iy_i)}
$$

> 3) Special circumstances:

$$
Cov(a+bx, c+dy) = bdCov(x, y)
$$

### 1.6.4 Correlation coefficient

The correlation coefficient is a quantity that studies the degree of linear correlation between variables. The correlation coefficient of two random variables is defined as: $$ Corr(x,y) = \\frac{Cov(x,y)}{\\sqrt{Var(x)Var(y)}} $$

> Properties of correlation coefficient:
> 1) Boundedness. The value range of the correlation coefficient is \[-1,1\], which can be regarded as a dimensionless covariance.
> 2) The closer the value is to 1, the stronger the positive correlation (linear) between the two variables. The closer it is to -1, the stronger the negative correlation. When it is 0, it means there is no correlation between the two variables.

## references

\[1\]Ian, Goodfellow, Yoshua, Bengio, Aaron...Deep Learning\[M\], People's Posts and Telecommunications Publishing, 2017

\[2\] Zhou Zhihua. Machine Learning\[M\]. Tsinghua University Press, 2016.

\[3\] Department of Mathematics, Tongji University. Advanced Mathematics (Seventh Edition) \[M\], Higher Education Press, 2014.

\[4\] Sheng Su, Shi Shiqian, Pan Chengyi, etc., eds. Probability Theory and Mathematical Statistics (4th Edition) \[M\], Higher Education Press, 2008
