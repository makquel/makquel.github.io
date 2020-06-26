---
layout: post
title:  "Probabilistic neural networks in a nutshell"
date:   2020-05-28 18:47:58 -0300
# categories: jekyll update
comments: true
mathjax: true
description: "Probabilistic neural network applied to classification and pattern recognition."
keywords: ""
---
<!-- Editar a resposta do https://stackoverflow.com/questions/14873203/plotting-of-1-dimensional-gaussian-distribution-function -->

<!-- https://jekyllrb.com/tutorials/using-jekyll-with-bundler/ -->
Probabilistic neural networks (PNN) are a type of feed-forward artificial neural network that are closely related to kernel density estimation (KDE) via Parzen-window that asymptotically approaches to Bayes optimal risk minimization. This technique is widely used to estimate class-conditional densities (also known as likelihood) in machine learning tasks such as supervised learning.

<figure>
  <img src="{{site.url}}/assets/img/pnn/pnn_architecture_git.png"/>
</figure>

The neural network that was introduced by [Specht](https://www.sciencedirect.com/science/article/abs/pii/089360809090049Q "Probabilistic neural networks") is composed by four layers: 
- Input layer: Features of data points (or observations)
- Pattern layer: Calculation of the class-conditional PDF
- Summation layer: Summation of the inter-class patterns
- Output layer: Hypothesis testing with the maximum a posteriori probability (MAP)
<!-- https://stackoverflow.com/questions/19331362/using-an-image-caption-in-markdown-jekyll -->
<!-- https://wordpress.com/support/markdown-quick-reference/ -->
<!-- http://cs.joensuu.fi/pages/oili/PR/?a=Some__Material&b=Linear__And__Nonlinear__Classifiers -->
<!-- https://math.stackexchange.com/questions/509465/standard-normal-random-variable-and-definition-of-phi -->
<!-- https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/ -->

In order to understand the back-bone mechanism of the PNN, one have to look back to Bayes theorem. Suppose that the goal is to is to built a Bayes classifier, where X and Θ are independent and identically distributed (i.i.d) random variables (r.v).

<figure align="center">
  <img src="{{site.url}}/assets/img/pnn/bayes_eq.png"/>
</figure>

whereas finding the likelihood probability density function (PDF) could be a challenging problem; using Parzen-window method to calculate it, tackles down this problem in a elegant and reliable way. Therefore, if the parameters of the likelihood PDF are known, it will be easy to infere the posterior probability.

The Parzen-window is, basically, a non-parametric method to estimate the PDF for a specific observation given a data set; conversely, this doesn't require prior knowledge about the underlying distribution. This window has a weighting funtion Φ and smoothing funtion h(n). (For further knowledge about KDE visit sebastian raschka [webpage](https://sebastianraschka.com/Articles/2014_kernel_density_est.html "Kernel density estimation via the Parzen-Rosenblatt window method"))

<!-- https://www.youtube.com/watch?v=MPaTYY-QnFw&t=47s -->
<!-- https://sebastianraschka.com/Articles/2014_kernel_density_est.html -->
<figure>
  <img src="{{site.url}}/assets/img/pnn/parzen-window.png"/>
</figure>

Using the normal distribution as weighting funtion lead us to the following equation, normalized by the total number of class conditional observations.
<figure>
  <img src="{{site.url}}/assets/img/pnn/likelihood_eq.png"/>
</figure>
In a multivariate problem Σ is a diagonal matrix that contains the covariance of each feature.
<figure>
  <img src="{{site.url}}/assets/img/pnn/cov_eq.png"/>
</figure>

For a better understanding, take for instance a simple univariate case study. Suppose that X is an i.i.d random variable that is composed by a set of binomial class data. Assume that σ=1, and a unclassified observation x=3. 
<figure>
  <img src="{{site.url}}/assets/img/pnn/X_normal_dist.png"/>
</figure>
<figure>
  <img src="{{site.url}}/assets/img/pnn/X_rv.png"/>
</figure>


Let Θ be a Bernoulli random variable that indicates the binomial class hypotheses, and let P(Θ) equaly likely. Under the hypothesis Θ=1, the random variable X has a PDF defined by:

<figure>
  <img src="{{site.url}}/assets/img/pnn/window_class_1.png"/>
</figure>

Under the alternative hypothesis Θ=2, X has a normal distribution with mean 2 and variance 1. 
<figure>
  <img src="{{site.url}}/assets/img/pnn/window_class_2.png"/>
</figure>
Therefore, a solution of x, that satisfies the boundary condition, can be found numerically. This is an optimal solution, that minimizes the misclassification rate. A proxy visual representation of the the hypothesis test of class conditional funtions is shown bellow.

<figure>
  <img src="{{site.url}}/assets/img/pnn/3d_example_pdfs.png"/>
</figure>

<figure>
  <img src="{{site.url}}/assets/img/pnn/2d_example_pdfs.png"/>
</figure>

The decision boundary of the PNN is given by:

<figure>
  <img src="{{site.url}}/assets/img/pnn/boundary_decision.png"/>
</figure>

The figure bellow shows the decision boundary and the error conditional probability (shaded region).

<figure>
  <img src="{{site.url}}/assets/img/pnn/decision_boundary.png"/>
</figure>

Finally, having observed x, is choosen an estimate that maximizes the posterior PDF ovel all Θ, via MAP.

<figure>
  <img src="{{site.url}}/assets/img/pnn/argmax.png"/>
</figure>

Given the MAP estimator, the outcome will be y2(x)=0.0011 < 0.2103 = y1(x), thus, the observation will be classified as Θ=1.

In order to compare with other machine learning algorithms, was created a python class that matches the structure of SciKit Learn algorithms. Using the default benchmark composed by 3 synthetic datasets was made a comparisson with a [Gaussian process](https://scikit-learn.org/stable/modules/gaussian_process.html ) and a [Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html ) classifiers.The image bellow shows the results achieved measured by the accuracy metric.

<!-- {% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')

{% endhighlight %} -->

<figure>
  <img src="{{site.url}}/assets/img/pnn/pnn_comparisson.png"/>
</figure>

Check out the [PNN](https://github.com/makquel/probabilistic-neural-network) repo for more info.


<!-- You’ll find this post in your `_posts` directory.

Jekyll requires blog post files to be named according to the following format:

`YEAR-MONTH-DAY-title.MARKUP`

Where `YEAR` is a four-digit number, `MONTH` and `DAY` are both two-digit numbers, and `MARKUP` is the file extension representing the format used in the file. After that, include the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://github.com/makquel/probabilistic-neural-network
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/ -->
