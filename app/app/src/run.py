import os
import json
from flask import Flask, render_template
from flask_bootstrap import Bootstrap

from config import Config

app = Flask(__name__)
app.config.from_object(Config)
bootstrap = Bootstrap(app)

with open('./machine_learning/tag_extraction/tag.json', 'r') as f:
    tag_list = json.load(f)
hair_color_list= tag_list['hair_color']
eye_color_list = tag_list['eye_color']

hair_color_choices = [(hair_color[:-5], hair_color[:-5]) for hair_color in hair_color_list]
eye_color_choices = [(eye_color[:-5], eye_color[:-5]) for eye_color in eye_color_list]

#model = hogehoge

from routes import *
    