import os
from io import BytesIO
import numpy as np

from PIL import Image
from flask import flash, render_template, request, redirect, url_for, send_from_directory
from flask_mail import Message

from run import app, mail, mail_info, encoder
from forms import IllustUploadForm, TagSelectForm, ContactForm
from recommender import Recommender

GET_IMG_NUM = 5
CANDIDATE_NUM = 500
FEATURE_PATH = './machine_learning/feature_extraction/feature'



@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')



@app.route('/about')
def about():
	return render_template('about.html')



@app.route('/contact', methods=['GET', 'POST'])
def contact():
	contact_form = ContactForm()
	if request.method == 'POST':
		if contact_form.validate() == False:
			return render_template('contact.html', form=contact_form)
		else:
			msg = Message(contact_form.subject.data, sender=mail_info['mail_username'], recipients=mail_info['mail_username'])
			msg.body = """
			From: %s &lt;%s&gt;
			%s
			""" % (contact_form.name.data, contact_form.email.data, contact_form.message.data)
			mail.send(msg)
			return 'Form posted'
	elif request.method == 'GET':
		return render_template('contact.html', form=contact_form)



@app.route('/illust_condition', methods=['GET', 'POST'])
def illust_condition():
	upload_form = IllustUploadForm()
	if upload_form.validate_on_submit():
		img = Image.open(BytesIO(upload_form.illust.data.read()))
		base_feature = encoder.extract_feature(img.resize((64, 64)))
		
		candidate_illust_names = Recommender.choose_illusts_from_tag(CANDIDATE_NUM, None, None)
		candidate_feature_paths = [os.path.join(FEATURE_PATH, candidate_illust_name.replace('png', 'npy')) for candidate_illust_name in candidate_illust_names]
		similarity_order = encoder.cal_similarity_order(GET_IMG_NUM, base_feature, candidate_feature_paths)
		
		illust_names = []
		for similarity_idx in similarity_order:
			illust_names.append(candidate_illust_names[similarity_idx])
		illust_paths = [os.path.join('./static/', illust_name) for illust_name in illust_names]

		if len(illust_paths) == 0:
			return render_template('no_illust.html')

		Recommender.stamp_used_img(illust_names)
		return render_template('show_illust.html', illust_paths=illust_paths)

	else:
		return render_template('illust_condition_input.html', form=upload_form)



@app.route('/tag_condition', methods=['GET', 'POST'])
def tag_condition():
	tag_select_form = TagSelectForm()
	if tag_select_form.validate_on_submit():
		hair_color = tag_select_form.hair_color_type.data
		eye_color = tag_select_form.eye_color_type.data
		illust_names = Recommender.choose_illusts_from_tag(GET_IMG_NUM, hair_color, eye_color)
		if len(illust_names) < GET_IMG_NUM:
			additional_illust_names = Recommender.choose_illusts_from_tag(GET_IMG_NUM - len(illust_names), hair_color, None)
			illust_names += additional_illust_names
		illust_paths = [os.path.join('./static/', illust_name) for illust_name in illust_names]

		if len(illust_paths) == 0:
			return render_template('no_illust.html')
			
		Recommender.stamp_used_img(illust_names)
		return render_template('show_illust.html', illust_paths=illust_paths)

	else:
		return render_template('tag_condition.html', form=tag_select_form)

