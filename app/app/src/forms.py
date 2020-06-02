from flask_wtf import FlaskForm
# from flask_wtf.file import FileField
from wtforms import BooleanField, SubmitField, FileField, ValidationError
from wtforms.validators import DataRequired

class IllustForm(FlaskForm):
    illust = FileField('Upload Illust')
    submit = SubmitField('Submit')

    def validate_illust(self, illust):
        if '.jpg' not in illust.data.filename and '.png' not in illust.data.filename:
            raise ValidationError('jpgまたはpng形式でアップロードして下さい')