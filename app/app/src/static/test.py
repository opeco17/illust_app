import glob
from PIL import Image

for img_path in glob.glob('./*.png'):
	img = Image.open(img_path)
	img = img.resize((128, 128))
	img.save(img_path)
