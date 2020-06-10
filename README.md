# Overview

<img align="right" src="https://user-images.githubusercontent.com/46510874/84046987-bf481a80-a9e5-11ea-9545-5788ad1c6ed0.gif" alt="Web Application Demo" width="450">

Generate your favorite anime character by multiple machine learning algorithms (GANs, Auto Encoder, I2V).

Activity contents are ...

1. Machine Learning Infrastracture
2. Machine Learning Algorithms
3. Web Application

What I painted in my hands can be seen [here](https://www.pixiv.net/users/44422398)!

# Machine Learning Infrastructure

<img width="1000" alt="スクリーンショット 2020-05-17 14 23 19" src="https://user-images.githubusercontent.com/46510874/82136546-1d039f80-984a-11ea-9cbb-5d7bb70450ec.png">

 - Use AWS for automaticaly and efficiently training models.
 - Most process are described by boto3 and its high level API (e.g. Step Functions Data Science SDK).
 
 - Codes -> app/
 
 # Machine Learning Algorithms
  - Use multiple meachine learning algorithms for generating images, recommender system, and tag extraction.
  
  ## SNGAN & WGAN-GP
  - Use SNGAN and WGAN-GP for generating high quality images.
 
 <img width="600" alt="スクリーンショット 2020-05-24 11 28 35" src="https://user-images.githubusercontent.com/46510874/82744354-2ba20780-9db2-11ea-88f5-865b93f26f6d.png">
  
   - Code -> ml_infra/machine_learning/sngan & ml_infra/machine_learning/wgan_gp
  
 ## Denoising Auto Encoder with Self Attention
  - Use Auto Encoder for image to image recommendation.
 
  - AutoEncoder extracts feature of images and calculate similarity between submitted ones and generated ones.
 
  - By using this model, I developed image based searching.
 
  - Code -> ml_infra/machine_learning/auto_encoder

## Illustration2Vec
 - Illustration2Vec is the VGG based image classifier.

 - This model can extract image tags like hair color, eye color and so on.

 - By using this model, I developed tag based searching.

 - Paper -> https://dl.acm.org/doi/abs/10.1145/2820903.2820907
 - Code -> https://github.com/rezoo/illustration2vec

# Web Application
<img width="986" alt="スクリーンショット 2020-05-31 19 58 41" src="https://user-images.githubusercontent.com/46510874/83350741-29731680-a379-11ea-8662-39e9e6e4faa3.png">
