import os
from flask import Flask, render_template
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
#model = hogehoge

from routes import *
