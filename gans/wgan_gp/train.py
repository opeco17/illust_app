import os
import glob
import math
import argparse
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn import init
from PIL import Image

import model
import loss
import dataloader


def train(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')

    img_path_list = glob.glob(args.data_dir + '/face_images/*/*.png')

    n_iter = 0
    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2
    train_epoch = args.train_epoch
    n_dis = args.n_dis
    batch_size = args.batch_size
    coef = args.coef

    gen_num_features = args.gen_num_features
    gen_dim_z = args.gen_dim_z
    gen_bottom_width = args.gen_bottom_width

    dis_num_features = args.dis_num_features

    wd_list = []

    gen = model.ResNetGenerator(gen_num_features, gen_dim_z, gen_bottom_width, F.relu).to(device)
    dis = model.ResNetDiscriminator(dis_num_features, F.relu).to(device)
        
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
    opt_dis = optim.Adam(dis.parameters(), lr=lr, betas=(beta1, beta2))

    dataset = dataloader.MyDataset(img_path_list)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)

    for epoch in range(train_epoch):
        print('Epoch: ', epoch)
        
        for batch in train_loader:
            if batch.shape[0] != batch_size:
                print('Skip')
                continue
                
            n_iter += 1
            print('n_iter: ', n_iter)
        
            for i in range(n_dis):
                if i == 0:
                    gen.zero_grad()
                    fake, _ = dataloader.sample_from_gen(batch_size, gen_dim_z, gen, device)
                    dis_fake = dis(fake).mean()
                    loss_gen = loss.cal_loss_gen(dis_fake)
                    loss_gen.backward()
                    opt_gen.step()
                
                dis.zero_grad()
                fake, _ = dataloader.sample_from_gen(batch_size, gen_dim_z, gen, device)
                real = batch.type(torch.float32).to(device)
                dis_fake, dis_real = dis(fake).mean(), dis(real).mean()
                gradient_penalty = loss.cal_gradient_penalty(dis, real, fake, coef, device)
                loss_dis = loss.cal_loss_dis(dis_fake, dis_real, gradient_penalty)
                loss_dis.backward()
                opt_dis.step()          

        if epoch % 1 == 0:
            save_model(gen, args.model_dir, epoch)


def save_model(gen, model_dir, epoch):
    path = os.path.join(model_dir, 'generator-{0}epoch.pth'.format(epoch))
    torch.save(gen.cpu().state_dict(), path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_epoch', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_dis', type=int, default=5)
    parser.add_argument('--coef', type=int, default=10)

    parser.add_argument('--gen_num_features', type=int, default=64)
    parser.add_argument('--gen_dim_z', type=int, default=128)
    parser.add_argument('--gen_bottom_width', type=int, default=4)
    parser.add_argument('--dis_num_features', type=int, default=64)

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num_gpus',type=int, default=os.environ['SM_NUM_GPUS'])


    train(parser.parse_args())
    