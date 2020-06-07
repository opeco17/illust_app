import os
import random
import glob

from db_connect import DBConnector


class Recommender(object):
    @classmethod
    def choose_illusts_from_tag(self, get_img_num, hair_color, eye_color):
        conn = DBConnector()
        img_names = conn.get_img_names_from_tag(get_img_num, hair_color, eye_color)
        conn.complete()
        random.shuffle(img_names)
        return img_names


    @classmethod
    def stamp_used_img(self, img_names):
        conn = DBConnector()
        conn.stamp_used_img(img_names)
        conn.complete()
