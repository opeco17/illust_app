# Overview
Generate my favorite anime character by machine learning (GANs).
Dataset are prepared by scraiping from pixiv users I bookmarked.

Activity contents are ...

1. Machine Learning Infrastracture development
2. Machine Learning model (GANs) development
3. Web Application development

# Machine Learning Infrastructure

<img width="1000" alt="スクリーンショット 2020-05-17 14 23 19" src="https://user-images.githubusercontent.com/46510874/82136546-1d039f80-984a-11ea-9cbb-5d7bb70450ec.png">

 - Use AWS for automaticaly and efficiently training models.
 - Most process are described by boto3 and its high level API (i.e. Step Functions Data Science SDK).
 
 # Machine Learning (GANs) 
  - Using GANs (Generative Adversarial Networks) for generating anime character images.
  - Try various GANs models and various preprocessing to generate best quality anime character.
  
  ## Dataset
   - Dataset consists of almost 30k my favorite users' works in [pixiv](https://www.pixiv.net/).
   - Images are scraiped by using [pixivpy](https://github.com/upbit/pixivpy), third party high level API.
   - Collected images are clipped to extract face by [OpenCV](https://opencv.org/).
   - Tags of images (e.g. hair colors, eyes colors...) are predicted by [illustration2vec](https://github.com/rezoo/illustration2vec), a CNN based deep learning model.
  
  ## cGANs with Projection Discriminator
  Paper -> https://arxiv.org/abs/1802.05637
  Code -> gans/projection_discriminator
  
 <img width="600" alt="スクリーンショット 2020-05-24 11 28 35" src="https://user-images.githubusercontent.com/46510874/82744354-2ba20780-9db2-11ea-88f5-865b93f26f6d.png">
 
 ## WGAN-GP
 Paper -> https://arxiv.org/abs/1704.00028
 
 
