import os
import json

import numpy as np
import pandas as pd
from PIL import Image
import boto3
from boto3 import Session

import i2v


def main():

    # Connect to S3
    bucket_name = 'pixiv-image-backet'
    client = boto3.client('s3')
    s3 = boto3.resource('s3')
    sbucket = Session().resource('s3').Bucket(bucket_name)
    keys = [obj.key for obj in sbucket.objects.filter(Prefix='origin_size_face_images')]

    client.download_file(bucket_name, 'image_tag.csv', './image_tag.csv')
    image_tag_df = pd.read_csv('image_tag.csv').drop('Unnamed: 0', axis=1)


    # Read hair color list
    with open('tag.json', 'r') as f:
        tag_list = json.load(f)
    hair_color_list = tag_list['hair_color']


    # Except completed images
    completed_image_list = ['origin_size_face_images/'+img_name for img_name in list(image_tag_df['image name'])]
    keys = list(set(keys) - set(completed_image_list))


    # Load illustration2vec
    illust2vec = i2v.make_i2v_with_chainer(
        './i2v/illust2vec_tag_ver200.caffemodel',
        './i2v/tag_list.json'
    )


    # Tag extraction
    for key in keys:
        print(key)
        image_name = key.lstrip('origin_size_face_images/')
        client.download_file(bucket_name, key, './tmp_image.png')
        image = Image.open('./tmp_image.png')
        result = illust2vec.estimate_plausible_tags([image], threshold=0.2)
        
        image_tag_df = image_tag_df.append(pd.Series([image_name, None], index=image_tag_df.columns), ignore_index=True)
        for tag, _ in result[0]['general']:
            if tag in hair_color_list:
                image_tag_df.loc[image_tag_df['image name']==image_name, 'hair color']=tag
                break        


    # Uploade to S3
    image_tag_df.to_csv('image_tag.csv')
    sbucket.upload_file('./image_tag.csv', 'image_tag.csv')


if __name__ == '__main__':
    main()