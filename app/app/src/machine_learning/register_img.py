import os
import sys
import json
import glob
import numpy as np
from PIL import Image

import tag_extraction
from feature_extraction import model
from feature_extraction import extract_feature

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)

import db_connect


PREVIOUS_IMG_DIR = './img_generator/classified_images/'
PROCESSED_IMG_DIR = './img_generator/processed_images/'
NEW_IMG_DIR = '../static/'


def scale_and_move_img(img, new_img_name):
    img.save(os.path.join(PROCESSED_IMG_DIR, new_img_name))
    resized_img = img.resize((128, 128))
    resized_img.save(os.path.join(NEW_IMG_DIR, new_img_name))


def extract_hair_color_and_eye_color(img, i2v):
    with open('./tag_extraction/tag.json', 'r') as f:
        tag_list = json.load(f)
    hair_color_list = tag_list['hair_color']
    eye_color_list = tag_list['eye_color']

    result = i2v.estimate_plausible_tags([img], threshold=0.05)
    hair_color = None
    eye_color = None
    for tag, _ in result[0]['general']:
        if hair_color is None and tag in hair_color_list:
            hair_color = tag[:-5]
        if eye_color is None and tag in eye_color_list:
            eye_color = tag[:-5]
        if hair_color is not None and eye_color is not None:
            continue
    return hair_color, eye_color


def main():
    # Load illustration2vec
    illust2vec = tag_extraction.make_i2v_with_chainer(
        './tag_extraction/illust2vec_tag_ver200.caffemodel',
        './tag_extraction/tag_list.json'
    )

    # Load Encoder 
    encoder = model.load_model('./feature_extraction/parameter')

    # Connect to DB
    conn = db_connect.DBConnector()
    last_img_name = conn.get_last_img_name()
    num = int(last_img_name.rstrip('.png'))

    img_path_list = glob.glob(os.path.join(PREVIOUS_IMG_DIR, '*.png'))

    print('Start')
    for img_path in img_path_list:
        num += 1
        new_img_name = str(num)+'.png'
        img = Image.open(img_path)
        hair_color, eye_color = extract_hair_color_and_eye_color(img, illust2vec)
        feature = extract_feature.extract(encoder, img)

        conn.insert_img_info(new_img_name, hair_color, eye_color)
        scale_and_move_img(img, new_img_name)
        np.save(os.path.join('feature', str(num)+'.npy'), feature)
        os.remove(img_path)

        if num == 1000:
            break
    conn.complete()


if __name__ == '__main__':
    main()
