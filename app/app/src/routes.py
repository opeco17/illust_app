import os
from flask import flash, render_template, request, redirect, url_for, send_from_directory
from run import app
from forms import IllustForm

@app.route('/')
@app.route('/index')
def index():
	print()
	return render_template('index.html')

@app.route('/illust_condition', methods=['GET', 'POST'])
def illust_condition():
	upload_form = IllustForm()
	if upload_form.validate_on_submit():
		print(os.listdir('./'))
		upload_form.illust.data.save('./upload_illust/'+'1.png')
		return render_template('illust_condition_input.html', form=upload_form)

	else:
		# if upload_form.illust.data is not None:
		# 	print(upload_form.illust.data.filename)
		return render_template('illust_condition_input.html', form=upload_form)


@app.route('/tag_condition', methods=['GET', 'POST'])
def tag_condition():
	return render_template('tag_condition.html')	