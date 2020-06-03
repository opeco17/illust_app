from flask_wtf import FlaskForm
from wtforms import BooleanField, SubmitField, FileField, ValidationError, RadioField, SelectField
from wtforms.validators import DataRequired

class IllustUploadForm(FlaskForm):
    illust = FileField('', validators=[ValidationError])
    submit = SubmitField('Submit')

    def validate_illust(self, illust):
        if '.jpg' not in illust.data.filename and '.png' not in illust.data.filename:
            raise ValidationError('jpgまたはpng形式でアップロードして下さい')


class TagSelectForm(FlaskForm):
    # sex = RadioField('Sex', choices=[(1,'Male'),(2,'Female')])
    hair_color_type = SelectField('Hair color', choices=[(1, 'Black'), (2, 'Pink')], coerce=int)
    eye_color_type = SelectField('Eye color', choices=[(1, 'Black'), (2, 'Pink')], coerce=int)
    submit = SubmitField('Submit')