import os
from flask import flash, render_template, request, redirect, url_for, send_from_directory

from run import app
from forms import IllustUploadForm, TagSelectForm
from recommender import IllustChooser

@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/illust_condition', methods=['GET', 'POST'])
def illust_condition():
	upload_form = IllustUploadForm()
	if upload_form.validate_on_submit():
		upload_form.illust.data.save('./upload_illust/'+'1.png')
		return render_template('show_illust.html')

	else:
		# if upload_form.illust.data is not None:
		# 	print(upload_form.illust.data.filename)
		return render_template('illust_condition_input.html', form=upload_form)


@app.route('/tag_condition', methods=['GET', 'POST'])
def tag_condition():
	tag_select_form = TagSelectForm()
	if tag_select_form.validate_on_submit():
		illust_ids = IllustChooser.choose_illust_ids()
		print(illust_ids)
		return render_template('show_illust.html')
	else:
		return render_template('tag_condition.html', form=tag_select_form)
