# Overview

<img align="right" src="https://user-images.githubusercontent.com/46510874/84046987-bf481a80-a9e5-11ea-9545-5788ad1c6ed0.gif" alt="Web Application Demo" width="550">

This application generates your favorite anime character by using  multiple machine learning algorithms (GANs, AutoEncoder, I2V).

## Tools

Machine Learning: PyTorch

Application: Flask / MySQL / NGINX / uWSGI / Docker

Infrastructure: AWS / Docker

 # Machine Learning Algorithms
  - Use multiple meachine learning algorithms for generating images, recommender system, and tag extraction.
  
  ## SNGAN
  - Use SNGAN (Spectral Normalized GAN) for generating high quality images.
 
 <img width="600" alt="スクリーンショット 2020-05-24 11 28 35" src="https://user-images.githubusercontent.com/46510874/82744354-2ba20780-9db2-11ea-88f5-865b93f26f6d.png">
  
   - Code -> ml_infra/machine_learning/sngan
  
 ## Auto Encoder
  - Use customized Auto Encoder model for image retrieval (image to image recommendation).
 
  - AutoEncoder extracts feature of image and calculates similarity between submitted ones and generated ones.
  
  <img width="700" alt="recommend_sample" src="https://user-images.githubusercontent.com/46510874/84335694-65577880-abd0-11ea-9526-1f405f7db912.png">
 
  - Code -> ml_infra/machine_learning/auto_encoder

## Illustration2Vec
 - Illustration2Vec is the VGG based image classifier to predict tags (hair color, eye color...).

 - This model enables tag based recommendation.

 - Paper -> https://dl.acm.org/doi/abs/10.1145/2820903.2820907
 
 - Code -> https://github.com/rezoo/illustration2vec
 
# Machine Learning Infrastructure

<img width="900" alt="スクリーンショット 2020-05-17 14 23 19" src="https://user-images.githubusercontent.com/46510874/82136546-1d039f80-984a-11ea-9cbb-5d7bb70450ec.png">

 - Use AWS for automaticaly and efficiently training models.
 - Most process are described with boto3 and its high level API (e.g. Step Functions Data Science SDK).
 
 - Codes -> app/

# Web Application
<img width="986" alt="スクリーンショット 2020-05-31 19 58 41" src="https://user-images.githubusercontent.com/46510874/83350741-29731680-a379-11ea-8662-39e9e6e4faa3.png">
