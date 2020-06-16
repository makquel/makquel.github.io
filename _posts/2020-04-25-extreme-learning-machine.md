---
layout: post
title:  "Introduction to extreme learning machine (ELM)"
date:   2020-04-25 17:47:58 -0300
# categories: jekyll update
comments: true
mathjax: true
description: "An introduction to Extreme learning machine (ELM)."
keywords: ""
---

<!-- https://jekyllrb.com/tutorials/using-jekyll-with-bundler/ -->
Extreme learning machine (ELM) is a supervised learning framework for simplifying the training process of Artificial neural networks (ANN). This framework was proposed by [Huang](https://www.ntu.edu.sg/home/egbhuang/ "Extreme learning machine"), as the name suggest, to provide better generalization performance at extreme fast learning speed. This work  showed … Single Hidden Layer Feedforward Neural Networks (SLFN). SLFNs are neural networks composed by only one hidden layer, where the output layer weights are updated during training. It has been exhaustively proved that multi-layer perceptron (MLP) with only a hidden layer can, sufficiently, approximate any continuous function giving origin to SLFNs, however, this does not guarantee optimal learning rate, generalization capabilities, and ease of implementation. 

<figure>
  <img src="{{site.url}}/assets/img/elm/elm_architecture_git.png"/>
</figure>

Compared to SLFN-ELM, traditional supervised learning methods have shown that learning speed of feedforward networks are slower than required. Most of this traditional methods are either gradient based (e.g. backpropagation algortihm) or evolutionary algorithms. The former, could be approached as unrestricted nonlinear optimization problem; where the bottleneck lies on the existence of several local minima, with the underlying assumption of multi-modal nature of the loss function; the latter are rather global optimization problems inspired by biological evolution.


The SLFN-ELM could be interpreted as linear system: 
<figure>
  <img src="{{site.url}}/assets/img/elm/linear_elm.png"/>
</figure>

<figure>
  <img src="{{site.url}}/assets/img/elm/huang_notation.png"/>
</figure>



So first, in order to check the algorithm that we just have implemented, a simple but usefull hard test is the XOR function, which is defined as a not linearly separable function.since it could overcome the limitations of linear separability...

<figure>
  <img src="{{site.url}}/assets/img/elm/XOR_eq.png"/>
</figure>


<figure>
  <img src="{{site.url}}/assets/img/elm/xor_elm_architecture_git.png"/>
</figure>