Title: Transfomer Literature Review


# 1. Introduction

# 2. Deep Learning

A neural network takes an input vector of $p$ variables $X=(X_1, X_2, \dots, X_p)$ and builds a non-linear function $f(X)$ to predict the response $Y$. A (single larger) neural network model as the form


The class of feedforward neural networks are based around a particular class of hypotheses $h: \mathcal{X} = \mathbb{R}^p \rightarrow \mathbb{R}$ with general form:

$$f(X) = A^{(d)} \circ g \circ A^{(d-1)} \circ g \circ \dots \circ g \circ A^{(2)} \circ g \circ A^{(1)}(x)$$

where:

- $d$ is known as the depth of the network;
-  $A^{(k)}(v) = \beta^{(k)}v + μ^{(k)}$ where $v ∈ \mathbb{R}^{m_k} , \beta^{(k)} ∈ \mathbb{R}^{m_{k+1}×m_k} , μ^{(k)} ∈ \mathbb{R}^{m_{k+1}}$ with $m_1 = p$
and $m_{d+1} = 1$;
- $g : \mathbb{R}^m → \mathbb{R}^m$ applies (for any given $m$) a so-called activation function $ψ : \mathbb{R} → \mathbb{R}$ elementwise i.e. for $v = (v_1, \dots , v_m)^T$ , $g(v) = (ψ(v_1), \dots , ψ(v_m))^T$ . The activation
function is nonlinear and typical choices include:
    - (i) $u  → \max(u, 0)$ (known as a rectified linear unit (ReLU))
    - (ii) $u  → 1/(1 + e−u)$ (sigmoid).


## 2.1. Building from single layer [Further explanation ]:

Neural networks are a powerful modelling approach that accounts for interactions especially well. A neural network takes an input vector of $p$ variables $X = (X_1,X_2, \dots, X_p)$ and builds a non-linear function $f(X)$ to predict the response $Y$. 

First consider the simple single layer neural layer, each node in this layer is represents "activations" - or outputs from the activation function. More precisely, we pre-specify a nonlinear function $$g$$ called the activation function. Then, at each node $k \in \{1, \dots, K\}$, we denote the weights  $w_{k0}, \dots, w_{kp}$ and compute the the activation for input features $X$ by:

$h_k(X) = g(w_{k0} + \sum_{j=1}^{p} w_{kj}X_j)$.

In view of matrix notation, typically the activation notation $h_k(X)$ is further simplified to 
$A_k$.

At each node, we may also add biases denoted $\beta_0, \dots, \beta_K$; Thus, the (single layer) neural network model has the form: 

$$\begin{aligned}
f(X) &:= \beta_0 + \sum_{k=1}^K\beta_k A_k\\
&= \beta_0 + \sum_{k=1}^{K}\beta_k h_k(X)\\
&= \beta_0 + \sum_{k=1}^{K}\beta_k g(w_{k0} + \sum_{j=1}^{p}w_{kj}X_j).
\end{aligned}$$

In summary, These $K$ activations from the hidden layer then feed into the output layer, resulting in 

$$f(X) = \beta_0 + \sum_{k=1}^{K} \beta_k A_{k},$$

a linear regression model in the $$K$$ activations. All the parameters $\beta_0, \dots, \beta_k$ and $w_{10}, \dots, w_{kp}$ need to be estimated from data.


# 3. Nueral Network for Sequential Data
## 3.1. Recurrent Neural Networks (RNN)

## 3.2. Rolling 

## 3.3. Long Short-Term Memmory Network

# 4. Attention and Transformers

## 4.1. Word-Embedding and Word2Vec

## 4.2. Positional Encoding

# 5. Applications in Finance, 

## 5.1. Deep Learning Applications

## 5.2. Transformer Applications