import sys
import torch
from flask import Flask, request, render_template
sys.path.append('../')

from anime_face_generator.projection_discriminator_64 import model
from generate import generate


app = Flask(__name__)
hair_colors = ['blue', 'red', 'green', 'black', 'white', 'pink', 'purple', 'brown', 'orange', 'silver', 'blonde']
hair_colors_dict = {hair_color: num for num, hair_color in enumerate(hair_colors)}
num_classes = len(hair_colors)
generator = model.ResNetGenerator(num_classes=num_classes)
generator.load_state_dict(torch.load('../anime_face_generator/projection_discriminator_64/gen_parameter', map_location=torch.device('cpu')))
generator.eval()
print('Model Loaded')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('input.html', hair_colors=hair_colors)

    if request.method == 'POST':
        num_imgs = 3
        file_urls = ['static/images/fake_img'+str(i)+'.png' for i in range(num_imgs)]
        hair_color = request.form['select']
        label = hair_colors_dict[hair_color]
        generate(generator, label, num_imgs, file_urls)
        return render_template('result.html', file_urls=file_urls, num_imgs=list(range(num_imgs)))

if __name__ == '__main__':
    app.run(debug=True)