import torch
import numpy as np
from PIL import Image

def generate(generator, label, img_num, file_urls):
    with torch.no_grad():
        input_z = torch.empty(img_num, 128, dtype=torch.float32, device='cpu').normal_()
        pseudo_labels = torch.from_numpy(np.ones((img_num))).type(torch.long).to('cpu')*label
        fake_imgs = generator(input_z, pseudo_labels)
        fake_imgs = fake_imgs.numpy().transpose(0, 2, 3, 1)
        fake_imgs = (0.5 * fake_imgs + 0.5)*255
        for i in range(img_num):
            fake_img = fake_imgs[i]
            fake_img = Image.fromarray(np.uint8(fake_img))
            fake_img.save(file_urls[i])
