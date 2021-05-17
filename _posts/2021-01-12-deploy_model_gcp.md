---
layout: post
title:  "Containerizing and deploying a machine learning model as a service [WIP]"
date:   2021-01-12 22:36:00 -0300
# categories: jekyll update
comments: true
mathjax: true
description: "."
keywords: ""
---



<figure>
  <img src="{{site.url}}/assets/img/deploy/fluxograma_finep-deploy_medium.png"/>
</figure>

One of the neatest things about Machine Learning projects is to be able to serve them as a service whether on-premise or cloud. When comes to deploy a containerized application on a fully managed serverless platform there is an affordable option (free up to 2M request/month) called Cloud Run. The usual workflow consists of two parts: first submitting(and versioning) your app's container, and finally deploy it to the platform.  The following shell script shows how to do it.

<figure>
  <img src="{{site.url}}/assets/img/deploy/fluxograma_finep-deploy_medium.png"/>
</figure>