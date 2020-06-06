import os
from io import BytesIO
import numpy as np

from PIL import Image
from flask import flash, render_template, request, redirect, url_for, send_from_directory

from run import app
from run import i2v
from run import encoder
from forms import IllustUploadForm, TagSelectForm
from recommender import IllustChooser


@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')


@app.route('/illust_condition', methods=['GET', 'POST'])
def illust_condition():
	get_img_num = 10
	upload_form = IllustUploadForm()
	if upload_form.validate_on_submit():
		# upload_form.illust.data.save(os.path.join('./upload_illust', 'uploaded.png'))
		# img = Image.open(os.path.join('./upload_illust', 'uploaded.png'))
		img = Image.open(BytesIO(upload_form.illust.data.read())).resize((64, 64))

		hair_color, eye_color = i2v.extract_hair_color_and_eye_color(img)
		illust_names = IllustChooser.choose_illusts_from_tag(300, hair_color, eye_color)
		illust_paths = [os.path.join('./static/', illust_name) for illust_name in illust_names]
		other_feature_paths = [os.path.join('./machine_learning/feature_extraction/feature', illust_name.replace('png', 'npy')) for illust_name in illust_names]
	
		feature = encoder.extract_feature(img)
		similarity_order = encoder.cal_similarity_order(get_img_num, feature, other_feature_paths)
		
		new_illust_paths = []
		for similarity_idx in similarity_order:
			new_illust_paths.append(illust_paths[similarity_idx])
		return render_template('show_illust.html', illust_paths_first=new_illust_paths[:5], illust_paths_second=new_illust_paths[5:])

	else:
		# if upload_form.illust.data is not None:
		# 	print(upload_form.illust.data.filename)
		return render_template('illust_condition_input.html', form=upload_form)


@app.route('/tag_condition', methods=['GET', 'POST'])
def tag_condition():
	get_img_num = 10
	tag_select_form = TagSelectForm()
	if tag_select_form.validate_on_submit():
		hair_color = tag_select_form.hair_color_type.data
		eye_color = tag_select_form.eye_color_type.data
		illust_names = IllustChooser.choose_illusts_from_tag(get_img_num, hair_color, eye_color)
		illust_paths = [os.path.join('./static/', illust_name) for illust_name in illust_names]
		return render_template('show_illust.html', illust_paths_first=illust_paths[:5], illust_paths_second=illust_paths[5:])
	else:
		return render_template('tag_condition.html', form=tag_select_form)
