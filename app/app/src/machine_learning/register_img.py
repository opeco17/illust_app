import os
import sys
import json
import argparse
import glob
import numpy as np
from PIL import Image

from tag_extraction.tag_extraction import TagExtractor
from feature_extraction import model

pardir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pardir)

import db_connect


PREVIOUS_IMG_DIR = './img_generator/raw_images/'
PROCESSED_IMG_DIR = './img_generator/processed_images/'
NEW_IMG_DIR = '../static/'
I2V_MODULE_PATH = './tag_extraction'
ENCODER_MODULE_PATH = './feature_extraction'


def move_img(img, new_img_name, hair_color):
    img.save(os.path.join(PROCESSED_IMG_DIR, hair_color, new_img_name))
    resized_img = img.resize((128, 128))
    resized_img.save(os.path.join(NEW_IMG_DIR, new_img_name))


def main(args):
    # Load Illustration2Vec
    i2v = TagExtractor(I2V_MODULE_PATH)

    # Load Encoder 
    encoder = model.load_model(os.path.join(ENCODER_MODULE_PATH, 'parameter'))

    # Connect to DB
    conn = db_connect.DBConnector()
    last_img_name = conn.get_last_img_name()
    num = int(last_img_name.rstrip('.png'))

    print('Start')
    for hair_color in i2v.hair_color_list:
        print(hair_color)
        os.mkdir(os.path.join(PROCESSED_IMG_DIR, hair_color)) if not os.path.exists(os.path.join(PROCESSED_IMG_DIR, hair_color)) else None

        img_path_list = glob.glob(os.path.join(PREVIOUS_IMG_DIR, hair_color, '*.png'))

        for i, img_path in enumerate(img_path_list):
            print(num)
            num += 1
            new_img_name = str(num)+'.png'
            img = Image.open(img_path)
            eye_color = i2v.extract_eye_color(img)
            feature = encoder.extract_feature(img)

            conn.insert_img_info(new_img_name, hair_color, eye_color)
            move_img(img, new_img_name, hair_color)
            np.save(os.path.join(ENCODER_MODULE_PATH, 'feature',str(num)+'.npy'), feature)
            # os.remove(img_path)

            if i == args.max_num:
                break

    conn.complete()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_num', type=int, default=0)

    main(parser.parse_args())
