import random
import os
import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

from model import AutoEncoder
from dataloader import MyDataset


def main():
    print('Start')
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train_epoch = 1000

    images_path = glob.glob('../64_size_face_images/*.png')
    images_path = random.sample(images_path, len(images_path))
    split_num = int(len(images_path)*0.8)
    train_path = images_path[:split_num]
    test_path = images_path[split_num:]
    result_path = images_path[-15:]

    train_dataset = MyDataset(train_path)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = MyDataset(test_path)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    result_dataset = MyDataset(result_path)
    result_dataloader = torch.utils.data.DataLoader(result_dataset, batch_size=result_dataset.__len__(), shuffle=False)
    result_images = next(iter(result_dataloader))

    model = AutoEncoder().to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    out_path = './output512-tanh'
    train_loss_list = []
    test_loss_list = []

    for epoch in range(train_epoch):
        model.to(device)
        loss_train = 0
        for x in train_dataloader:
            recon_x = model(x)
            loss = criterion(recon_x, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        loss_train /= train_dataloader.__len__()
        train_loss_list.append(loss_train)    

        if epoch % 1 == 0: 
            with torch.no_grad():
                model.eval()
                loss_test = 0
                for x_test in test_dataloader:
                    recon_x_test = model(x_test)
                    loss_test += criterion(recon_x_test, x_test).item()
                loss_test /= test_dataloader.__len__()
                test_loss_list.append(loss_test)
                np.save(os.path.join(out_path, 'train_loss.npy'), np.array(train_loss_list))
                np.save(os.path.join(out_path, 'test_loss.npy'), np.array(test_loss_list))
                model.train()
                
        
if __name__ == '__main__':
    main()