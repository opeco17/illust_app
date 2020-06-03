# Overview
Generate your favorite anime character by multiple machine learning models (GANs, AutoEncoder, CNN...).

Activity contents are ...

1. Machine Learning Infrastracture development
2. Machine Learning technology
3. Web Application development

What I painted can be seen [here](https://www.pixiv.net/users/44422398)!

# Machine Learning Infrastructure

<img width="1000" alt="スクリーンショット 2020-05-17 14 23 19" src="https://user-images.githubusercontent.com/46510874/82136546-1d039f80-984a-11ea-9cbb-5d7bb70450ec.png">

 - Use AWS for automaticaly and efficiently training models.
 - Most process are described by boto3 and its high level API (e.g. Step Functions Data Science SDK).
 
 # Machine Learning technology
  - Use multiple meachine learning models for generating images, recommender system, and tag extraction.
  
  ## Dataset
   - Dataset consists of almost 30k my favorite users' works in [pixiv](https://www.pixiv.net/).
   - Images are scraiped by using [pixivpy](https://github.com/upbit/pixivpy), third party high level API.
   - Collected images are clipped to extract face by [OpenCV](https://opencv.org/).
   - Tags of images (e.g. hair colors, eyes colors...) are predicted by [illustration2vec](https://github.com/rezoo/illustration2vec), a CNN based deep learning model.
  
  ## SNGAN
 Use SNGAN for generating high quality images.
 
 <img width="600" alt="スクリーンショット 2020-05-24 11 28 35" src="https://user-images.githubusercontent.com/46510874/82744354-2ba20780-9db2-11ea-88f5-865b93f26f6d.png">
  
  Code -> gans/sngan
  
  ## WGAN-GP
  It's not used now due to its ouput quality.
  
  Code -> gans/wgan
 
 ## Auto Encoder with Self Attention
 Use Auto Encoder for image to image recommendation system.
 
 AutoEncoder extracts feature of images and calculate similarity between submitted ones and generated ones.

# Web Application
## Web Application Infrastructure
<img width="986" alt="スクリーンショット 2020-05-31 19 58 41" src="https://user-images.githubusercontent.com/46510874/83350741-29731680-a379-11ea-8662-39e9e6e4faa3.png">

## Web Application Page
<img width="800" alt="スクリーンショット 2020-06-03 23 19 25" src="https://user-images.githubusercontent.com/46510874/83648170-b247b780-a5f0-11ea-950b-03def65fc3ac.png">

