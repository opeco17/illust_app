import os
from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from config import Config

app = Flask(__name__)
app.config.from_object(Config)
bootstrap = Bootstrap(app)
#model = hogehoge

from routes import *
