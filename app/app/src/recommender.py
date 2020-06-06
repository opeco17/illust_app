import os
import random
import glob

from db_connect import DBConnector


class IllustChooser(object):
    @classmethod
    def choose_illusts_from_tag(self, get_img_num, hair_color, eye_color):
        conn = DBConnector()
        img_names = conn.get_img_names_from_tag(get_img_num, hair_color, eye_color)
        conn.stamp_used_img(img_names)
        conn.complete()
        random.shuffle(img_names)
        return img_names


