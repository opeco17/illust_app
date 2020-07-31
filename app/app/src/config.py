import os
import json

class Config(object):
    with open('mail_info.json', 'r') as f:
        mail_info = json.load(f)
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    MAIL_SERVER = mail_info['mail_server']
    MAIL_PORT = 465
    MAIL_USE_SSL = True
    MAIL_USERNAME = mail_info['mail_username']
    MAIL_PASSWORD = mail_info['mail_password']
    DEBUG = False 


