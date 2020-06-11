import os
import glob

import numpy as np
from PIL import Image

import model

def main():
    encoder = model.load_model('./parameter')
    encoder.eval()

    feature_name_list = os.listdir('./feature')
    feature_num_list = [feature_name.rstrip('.npy') for feature_name in feature_name_list if 'npy' in feature_name]
    
    for feature_num in feature_num_list:
        img_name = feature_num + '.png'
        img_path = '../../static/' + img_name
        img = Image.open(img_path).resize((64, 64))
        feature = encoder.extract_feature(img)
        np.save('./feature/'+feature_num+'.npy', feature)

    print('Completed!!')        

if __name__ == '__main__':
    main()
