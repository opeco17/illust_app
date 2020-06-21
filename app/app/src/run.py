import os
import json
from flask import Flask, render_template
from flask_bootstrap import Bootstrap

from config import Config
from machine_learning.tag_extraction.tag_extraction import TagExtractor
from machine_learning.feature_extraction.model import load_model

I2V_MODULE_PATH = './machine_learning/tag_extraction'
ENCODER_MODULE_PATH = './machine_learning/feature_extraction'

app = Flask(__name__)
app.config.from_object(Config)
bootstrap = Bootstrap(app)
# i2v = TagExtractor(I2V_MODULE_PATH)
encoder = load_model(os.path.join(ENCODER_MODULE_PATH, 'parameter'))

with open(os.path.join(I2V_MODULE_PATH, 'tag.json'), 'r') as f:
    tag_list = json.load(f)
hair_color_list = tag_list['hair_color']
eye_color_list = tag_list['eye_color']

hair_color_choices = [(hair_color, hair_color[:-5].capitalize()) for hair_color in hair_color_list]
eye_color_choices = [(eye_color, eye_color[:-5].capitalize()) for eye_color in eye_color_list]

from routes import *
