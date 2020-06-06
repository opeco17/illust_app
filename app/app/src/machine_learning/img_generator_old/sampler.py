import numpy as np
import torch


def sample_z(batch_size, dim_z, device):
    return torch.empty(batch_size, dim_z, dtype=torch.float32, device=device).normal_()

def sample_pseudo_labels(num_classes, batch_size, device):
    pseudo_labels = torch.from_numpy(np.random.randint(low=0, high=num_classes, size=(batch_size)))
    pseudo_labels = pseudo_labels.type(torch.long).to(device)
    return pseudo_labels

def sample_pseudo_labels_const(num_classes, batch_size, label, device):
    pseudo_labels = torch.ones(batch_size)*label
    pseudo_labels = pseudo_labels.type(torch.long).to(device)
    return pseudo_labels
    
def sample_from_gen(num_classes, batch_size, dim_z, label, device, gen):
    if label == None:
        pseudo_y = sample_pseudo_labels(num_classes, batch_size, device)
    else:
        pseudo_y = sample_pseudo_labels_const(num_classes, batch_size, label, device)
    z = sample_z(batch_size, dim_z, device)
    fake = gen(z, pseudo_y) 
    fake_img = fake.cpu().detach().numpy().transpose(0, 2, 3, 1)
    fake_img = 0.5 * fake_img + 0.5
    pseudo_y = pseudo_y.cpu().detach().numpy()
    return fake_img, pseudo_y