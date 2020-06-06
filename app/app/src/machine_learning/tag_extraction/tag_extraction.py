import os
import json

from .chainer_i2v import make_i2v_with_chainer


class TagExtractor(object):
    
    def __init__(self, module_path):
        self.illust2vec = make_i2v_with_chainer(
            os.path.join(module_path, 'illust2vec_tag_ver200.caffemodel'),
            os.path.join(module_path, 'tag_list.json')
        )
        with open(os.path.join(module_path, 'tag.json'), 'r') as f:
            tag_list = json.load(f)
        self.hair_color_list = tag_list['hair_color']
        self.eye_color_list = tag_list['eye_color']


    def extract_eye_color(self, img):
        result = self.illust2vec.estimate_plausible_tags([img], threshold=0.05)
        eye_color = None
        for tag, _ in result[0]['general']:
            if tag in self.eye_color_list:
                eye_color = tag
                break
        return eye_color
        

    def extract_hair_color_and_eye_color(self, img):
        result = self.illust2vec.estimate_plausible_tags([img], threshold=0.05)
        hair_color = None
        eye_color = None
        for tag, _ in result[0]['general']:
            if hair_color is None and tag in self.hair_color_list:
                hair_color = tag
            if eye_color is None and tag in self.eye_color_list:
                eye_color = tag
            if eye_color is not None and hair_color is not None:
                break
        return hair_color, eye_color
        
