---
layout: post
title:  "Probabilistic neural networks in a nutshell"
date:   2020-12-25 07:45:00 -0300
# categories: jekyll update
comments: true
mathjax: true
description: "Processing medical images for deep learning applications."
keywords: ""
---

Image preprocessing is a fundamental step in any deep learning model building process, especialy when it comes to medical images that we heavely rely on such as X-ray and computed tomography(CT). Whether you are new to image processing or you have some experience, this is an overview of the challenges that may be faced when dealing with such images and how to overcome some of the commom pitfalls. From reading raw DICOM files and anonimizing them to preparing data to tensor of the input layer or even preparing data for radiomics analisis, this post is uses SimpleITK to achive such tasks. SimpleITK is a procedural ITK's wrapper for python language that has many bulidins from ITK popular package.

<b>Intro</b> 
Several machine learning and deep learning aplications using medical images still realy in some tecnlogies such as X-ray and computed tomography (CT) for desease diagnosis and prognosis. Build a suceesful data set using this images depend in image quality aspects such as signal-tonoise ratio and intesity homogenities to perform in reansonable manner. Dealing with inhomengities is a key aspect, since homogeneus datasets could perfom better during the training stage of the model. 

<figure>
  <img src="{{site.url}}/assets/img/img_proc/pre_process_pipeline.png"/>
</figure>


Let's take for instance ...