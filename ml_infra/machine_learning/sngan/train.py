import os
import glob
import json
import argparse
import pandas as pd

import torch
import torch.nn.functional as F
import torch.optim as optim

import dataloader
import model


def train(args):
    print('Start')

    if torch.cuda.is_available():
        device = 'cuda'
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = 'cpu'

    hair_colors = ["blonde hair", "brown hair", "black hair", "blue hair", "pink hair", "purple hair", "green hair", "red hair", "silver hair", "white hair", "orange hair", "aqua hair", "grey hair"]

    image_tag_df = pd.read_csv(os.path.join(args.data_dir, 'image_tag.csv'))
    image_tag_df = image_tag_df.dropna(how='any')    
    
    all_img_path_list = glob.glob(args.data_dir+'/face_images/*/*.png')
    all_img_name_list = [all_img_path.lstrip(args.data_dir+'/face_images/') for all_img_path in all_img_path_list]

    img_name_list = list(set(image_tag_df['image name']) & set(all_img_name_list))
    img_path_list = [args.data_dir+'/face_images/'+img_name for img_name in img_name_list]
  
    n_iter = 0
    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2
    train_epoch = args.train_epoch
    n_dis = args.n_dis
    batch_size = args.batch_size
    num_classes = len(hair_colors)

    gen_num_features = args.gen_num_features
    gen_dim_z = args.gen_dim_z
    gen_bottom_width = args.gen_bottom_width
    gen_distribution = args.gen_distribution

    dis_num_features = args.dis_num_features

    dataset = dataloader.MyDataset(img_name_list, img_path_list, image_tag_df, hair_colors)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    
    gen = model.ResNetGenerator(
        gen_num_features, gen_dim_z, gen_bottom_width,
        activation=F.relu, num_classes=num_classes
    ).to(device)

    dis = model.SNResNetProjectionDiscriminator(dis_num_features, num_classes, F.relu).to(device)
    
    opt_gen = optim.Adam(gen.parameters(), lr, (beta1, beta2))
    opt_dis = optim.Adam(dis.parameters(), lr, (beta1, beta2))
    
    gen_criterion = model.GenLoss()
    dis_criterion = model.DisLoss()

    for epoch in range(train_epoch):
        print('Epoch : ', epoch)
        
        for x_batch, y_batch in train_loader:
            n_iter += 1
            print('n_iter : ', n_iter)

            for i in range(n_dis):
                if i == 0:
                    fake, pseudo_y, _ = dataloader.sample_from_gen(num_classes, batch_size, gen_dim_z, device, gen)
                    dis_fake = dis(fake, pseudo_y)
                    loss_gen = gen_criterion(dis_fake, None)

                    gen.zero_grad()
                    loss_gen.backward()
                    opt_gen.step()

                fake, pseudo_y, _ = dataloader.sample_from_gen(num_classes, batch_size, gen_dim_z, device, gen)
                real, y = x_batch.type(torch.float32).to(device), y_batch.to(device)
                dis_fake, dis_real = dis(fake, pseudo_y), dis(real, y)
                loss_dis = dis_criterion(dis_fake, dis_real)

                dis.zero_grad()
                loss_dis.backward()
                opt_dis.step()

        if epoch % 1 == 0:
            save_model(gen, args.model_dir, epoch)


def save_model(gen, model_dir, epoch):
    path = os.path.join(model_dir, 'generator-{0}epoch.pth'.format(epoch))
    torch.save(gen.cpu().state_dict(), path)


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--train_epoch', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_dis', type=int, default=5)

    # Model Parameters
    parser.add_argument('--gen_num_features', type=int, default=64)
    parser.add_argument('--gen_dim_z', type=int, default=128)
    parser.add_argument('--gen_bottom_width', type=int, default=4)
    parser.add_argument('--gen_distribution', type=str, default='normal')
    parser.add_argument('--dis_num_features', type=int, default=64)

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num_gpus',type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())
