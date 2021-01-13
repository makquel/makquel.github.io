---
layout: post
title:  "Processing medical images for deep learning applications"
date:   2020-12-25 07:45:00 -0300
# categories: jekyll update
comments: true
mathjax: true
description: "Processing medical images for deep learning applications."
keywords: ""
---

Image preprocessing is a fundamental step in any deep learning model building process, especially when it comes to medical images that we heavily rely on such as X-ray and computed tomography(CT). Whether you are new to image processing or you have some experience, this is an overview of the challenges that may be faced when dealing with such images and how to overcome some of the common pitfalls. From reading raw DICOM files and anonymizing them to assembly tensor data of the input layer or even preparing data for radiomics analisis, this post uses SimpleITK to achieve such tasks. SimpleITK is a procedural ITK's wrapper for python language that has many bindings from ITK popular package.

<figure>
  <img src="{{site.url}}/assets/img/img_proc/ct_scans.png"/>
</figure>

<h1>Intro</h1> 

Several machine learning and deep learning applications using medical images still rely on some technologies such as X-ray and computed tomography (CT) for disease diagnosis and prognosis. Building a successful data set using these images depends on image quality aspects such as signal-to noise ratio and intensity homogeneities to perform in a reasonable manner. Dealing with inhomengities is a key aspect, since homogeneous datasets could perform better during the training stage of the model. Another important aspect is the link between existing health care systems and medical applications that should be in compliance with privacy regulations.

<figure>
  <img src="{{site.url}}/assets/img/img_proc/pre_process_pipeline.png"/>
</figure>

<h1>Anonymization</h1> 
Anonymizing and de-identifying patient data should be the first step in any medical application pipeline, since according to the newest LGP regulations, sensible data should be avoided when sharing datasets among the internet. These privacy regulations must be in compliance with organizations such as [HIPAA](https://www.hhs.gov/hipaa/index.html) in the US and the [PIPEDA](https://www.priv.gc.ca/en/privacy-topics/privacy-laws-in-canada/the-personal-information-protection-and-electronic-documents-act-pipeda/) in Canada. Recently the Radiological Society of North America(RSNA) made available an anonymization [tool](https://www.rsna.org/-/media/Files/RSNA/Covid-19/RICORD/RSNA-Anonymizer-Program-Instructions.pdf) for such purpose. 

[DICOM](https://dicom.innolitics.com/ciods/ct-image) objects for submission to the ...

<h1>Windowing</h1> 

<figure>
  <img src="{{site.url}}/assets/img/img_proc/level_intensities"/>
</figure>

<figure>
  <img src="{{site.url}}/assets/img/img_proc/level_intesities_comparison"/>
</figure>


Let's take for instance ...

<h1>Pseudo color converison</h1> 

Usually consolidated deep learning architectures for medical images have multiband like tensors that normally rely on RGB images. When using gray scale images (or even Hounsfield scale) comes in handy the use of pseudo color techniques that may enhance the target features within. This works whether you are using channels_first(NCHW) or channels_last (NHWC) conventions in TensorFlow for instance. The pseudocode (image below) shows how this process works.

<figure>
  <img src="{{site.url}}/assets/img/img_proc/red_blue_conversion"/>
</figure>

The code bellow implements this idea creating an multi band image in <b>YCbCr</b> color space for deep learning purposes. 

```python
def grey_to_color(image):
    """
    Converts an image array from grayscale (3 stacked channels) to YCbCr
    
    image:  gray scale image(w, h, 3)
  
    """
    R_channel = []
    G_channel = []
    B_channel = []
    ## Create LUT Red-Blue table
    H = pow(2,8)
    for elt in range(0,H):
        # lut_x = np.append(lut_x, np.floor(GrayScaleToBlueToRedColor(elt,255)).astype('uint8'), axis=0)
        R,G,B = np.floor(GrayScaleToBlueToRedColor(elt,H-1)).astype('uint8')
        R_channel.append(R)
        G_channel.append(G)
        B_channel.append(B)

    R_channel = np.asarray(R_channel)
    G_channel = np.asarray(G_channel)
    B_channel = np.asarray(B_channel)

    lut = np.dstack((B_channel, G_channel, R_channel))
    image = cv2.LUT(image, lut)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    return image
```