---
layout: post
title:  "Multi-step time series forecasting using Long short-term memory"
date:   2020-06-15 17:47:58 -0300
# categories: jekyll update
comments: true
mathjax: true
description: "Long short-term memory (LSTM) applied to multi-step forecast."
keywords: ""
---

<!-- https://jekyllrb.com/tutorials/using-jekyll-with-bundler/ -->

<figure>
  <img src="{{site.url}}/assets/img/lsmt/cover.png"/>
</figure>

## Intro 
<figure>
  <img src="{{site.url}}/assets/img/lsmt/pluviomentros_sub_rio_grande.png"/>
</figure>
Figure 1. Region of interest with rain gauges highlighted


## Exploratory data analysis (EDA)

<figure>
  <img src="{{site.url}}/assets/img/lsmt/TS_2015_2018.png"/>
</figure>
Figure 2. Seasonal hydraulic flow 

<figure>
  <img src="{{site.url}}/assets/img/lsmt/chuva_2015_2018.png"/>
</figure>
Figure 3. Seasonal rains within the region of interest  

<figure>
  <img src="{{site.url}}/assets/img/lsmt/corr_mat_chuva_vazao.png"/>
</figure>
Figure 4. Correlation matrix of the lagged variables
