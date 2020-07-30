import random
import os
import glob
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

from model import AutoEncoder
from dataloader import MyDataset


def add_noise(x, var):
    batch_size, channel, height, width = x.size()
    noise = torch.normal(0, var, (batch_size, channel, height, width))
    return x + noise


def train(args):
    print('Start')
    if torch.cuda.is_available():
        device = 'cuda'
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = 'cpu'

    train_epoch = args.train_epoch
    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2
    batch_size = args.batch_size
    noise_var = args.noise_var

    h_dim = args.h_dim

    images_path = glob.glob(args.data_dir+'/face_images/*/*.png')
    random.shuffle(images_path)
    split_num = int(len(images_path)*0.8)
    train_path = images_path[:split_num]
    test_path = images_path[split_num:]
    result_path = images_path[-15:]

    train_dataset = MyDataset(train_path)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MyDataset(test_path)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    result_dataset = MyDataset(result_path)
    result_dataloader = torch.utils.data.DataLoader(result_dataset, batch_size=result_dataset.__len__(), shuffle=False)
    result_images = next(iter(result_dataloader))

    model = AutoEncoder(h_dim=h_dim).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr, (beta1, beta2))

    out_path = args.model_dir
    train_loss_list = []
    test_loss_list = []

    for epoch in range(train_epoch):
        model.to(device)
        loss_train = 0
        for x in train_dataloader:
            noised_x = add_noise(x, noise_var)
            recon_x = model(noised_x)
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
    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--train_epoch', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--noise_var', type=float, default=0.1)

    # Model Parameters
    parser.add_argument('--h_dim', type=int, default=256)

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num_gpus',type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())