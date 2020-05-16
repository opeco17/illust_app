import os
from glob import glob

import numpy as np
import cv2
from PIL import Image
import boto3
from boto3 import Session


def main():
    # Connection to S3
    bucket_name = 'pixiv-image-backet'
    client = boto3.client('s3')
    s3 = boto3.resource('s3')
    sbucket = Session().resource('s3').Bucket(bucket_name)
    keys = [obj.key for obj in sbucket.objects.filter(Prefix='raw_images/')]

    for key in keys:
        print(key)
        # Download images
        image_name = key.lstrip('raw_images/').rstrip('.jpg')
        client.copy_object(Bucket=bucket_name, Key='processed_raw_images/'+image_name+'.jpg', CopySource={'Bucket': bucket_name, 'Key': key})
        client.download_file(bucket_name, key, './tmp_image.png')
        s3.Object(bucket_name, key).delete()
        origin_img = cv2.imread('./tmp_image.png', cv2.IMREAD_COLOR)
        
        if origin_img is None:
            print('No file found')
        else:
            # Face detection
            img = origin_img.copy()
            cascade_file = 'lbpcascade_animeface.xml'
            cascade = cv2.CascadeClassifier(cascade_file)
            img_face = img

            img_gray = cv2.cvtColor(img_face, cv2.COLOR_RGB2GRAY)
            face_list = cascade.detectMultiScale(img_gray, minSize=(100, 100))

            if len(face_list) == 0:
                print("No face detected")
                
            else:
                idx = 0
                for (x, y, w, h) in face_list:
                    # Face clipping
                    idx += 1
                    img_face_ex = origin_img[y-15:y+h+15, x-15:x+w+15]
                    img_color = cv2.cvtColor(img_face_ex, cv2.COLOR_BGR2RGB)

                    pil_img = Image.fromarray(img_color)
                    pil_img_resize = pil_img.resize((64, 64))
                    
                    pil_img.save('./origin_size_image.png')
                    pil_img_resize.save('./image.png')
                    sbucket.upload_file('./origin_size_image.png', 'origin_size_face_images/'+image_name+str(idx)+'.png')
                    sbucket.upload_file('./image.png', 'face_images/'+image_name+str(idx)+'.png')
                    

if __name__ == '__main__':
    main()