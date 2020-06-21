import json
from flask_wtf import FlaskForm
from wtforms import BooleanField, SubmitField, FileField, ValidationError, RadioField, SelectField
from wtforms.validators import DataRequired

from run import hair_color_choices, eye_color_choices

class IllustUploadForm(FlaskForm):
    illust = FileField('', validators=[ValidationError])
    submit = SubmitField('Submit')

    def validate_illust(self, illust):
        if '.jpg' not in illust.data.filename and '.png' not in illust.data.filename:
            raise ValidationError('jpgまたはpng形式でアップロードして下さい')


class TagSelectForm(FlaskForm):
    hair_color_type = SelectField('Hair Color', choices=hair_color_choices)
    eye_color_type = SelectField('Eye Color', choices=eye_color_choices)
    submit = SubmitField('Submit')
