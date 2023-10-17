import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import glob
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import warnings
import itertools
from PIL import Image
import torchvision.transforms as transforms
import argparse

# Define an ArgumentParser to accept command-line arguments
parser = argparse.ArgumentParser(description="CycleGAN Training Script")

# Add command-line arguments
parser.add_argument("--n_cpu", type=int, default=2, help="Number of CPU threads to use during batch generation")
parser.add_argument("--dataset_name", type=str, default="CycleGAN_data", help="Name of the dataset")
parser.add_argument("--root", type=str, default="data/CycleGAN_data", help="Root directory for the dataset")
parser.add_argument("--img_height", type=int, default=512, help="Height of images")
parser.add_argument("--img_width", type=int, default=512, help="Width of images")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--epoch", type=int, default=0, help="Epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs for training")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for Adam optimizer")
parser.add_argument("--b1", type=float, default=0.5, help="Adam: decay of first-order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="Adam: decay of second-order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=3, help="Epoch to start learning rate decay")

# Parsing command-line arguments
args = parser.parse_args()

# Use the args object to access these values throughout your code
n_cpu = args.n_cpu
dataset_name = args.dataset_name
root = args.root
img_height = args.img_height
img_width = args.img_width
channels = args.channels
epoch = args.epoch
n_epochs = args.n_epochs
batch_size = args.batch_size
lr = args.lr
b1 = args.b1
b2 = args.b2
decay_epoch = args.decay_epoch

# Applying CUDA to the above variables
cuda = torch.cuda.is_available()

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

# Defining our weights
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

# Applying the weights to our variables
G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

# Defining our optimizer
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2)
)

optimizer_D_A = torch.optim.Adam(
    D_A.parameters(), lr=lr, betas=(b1, b2)
)
optimizer_D_B = torch.optim.Adam(
    D_B.parameters(), lr=lr, betas=(b1, b2)
)

# Setting the number of epochs, delay, and offset
class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
        
    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

n_epochs = 10
epoch = 0
decay_epoch = 5

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G,
    lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)

lr_scheduler_D_A = torch.optim.lr.scheduler.LambdaLR(
    optimizer_D_A,
    lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B,
    lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
)

# Defining the transforms applied to the data
transforms_ = [
    transforms.Resize(int(img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]

# Saving the image as RGB
def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

# Getting the images and transforming them for training
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.mode = mode
        if self.mode == 'train':
            self.files_A = sorted(glob.glob(root + '/edges/*.png'))[:250]
            self.files_B = sorted(glob.glob(root + '/real/*.png'))[:250]
        elif self.mode == 'test':
            self.files_A = sorted(glob.glob(root + '/edges/*.png'))[:250]
            self.files_B = sorted(glob.glob(root + '/real/*.png'))[250:301]

    def  __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        
        if self.unaligned:
            image_B = Image.open(self.files_B[np.random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])
        if image_A.mode != 'RGB':
            image_A = to_rgb(image_A)
        if image_B.mode != 'RGB':
            image_B = to_rgb(image_B)
            
        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {'A': item_A, 'B': item_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
        
# Defining our dataloader for training and validation
dataloader = DataLoader(
    ImageDataset(root, transforms_=transforms_, unaligned=True),
    batch_size=1,
    shuffle=True,
    num_workers=0
)

val_dataloader = DataLoader(
    ImageDataset(root, transforms_=transforms_, unaligned=True, mode='test'),
    batch_size=5,
    shuffle=True,
    num_workers=0
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Training the model on our dataset
for epoch in range(epoch, n_epochs):
    for i, batch in enumerate(tqdm(dataloader)):
        
        # Set model input
        real_A = batch['A'].type(Tensor)
        real_B = batch['B'].type(Tensor)
        
        # Adversarial ground truths
        valid = Tensor(np.ones((real_A.size(0), *D_A.output_shape))) # requires_grad = False. Default.
        fake = Tensor(np.zeros((real_A.size(0), *D_A.output_shape))) # requires_grad = False. Default.

# Training the model on our dataset
for epoch in range(epoch, n_epochs):
    for i, batch in enumerate(tqdm(dataloader)):
        
        # Set model input
        real_A = batch['A'].type(Tensor)
        real_B = batch['B'].type(Tensor)
        
        # Adversarial ground truths
        valid = Tensor(np.ones((real_A.size(0), *D_A.output_shape)))
        fake = Tensor(np.zeros((real_A.size(0), *D_A.output_shape)))
        
        # Train Generators
        G_AB.train()
        G_BA.train()
        optimizer_G.zero_grad()
        
        # Identity Loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2
        
        # GAN Loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
        
        # Cycle Loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
        
        # Total Loss
        loss_G = loss_GAN + (10.0 * loss_cycle) + (5.0 * loss_identity)
        loss_G.backward()
        optimizer_G.step()
        
        # Train Discriminator A
        optimizer_D_A.zero_grad()
        loss_real = criterion_GAN(D_A(real_A), valid)
        loss_fake = criterion_GAN(D_A(fake_A.detach()), fake)
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        optimizer_D_A.step()
        
        # Train Discriminator B
        optimizer_D_B.zero_grad()
        loss_real = criterion_GAN(D_B(real_B), valid)
        loss_fake = criterion_GAN(D_B(fake_B.detach()), fake)
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.backward()
        optimizer_D_B.step()
        
        # Total Loss
        loss_D = (loss_D_A + loss_D_B) / 2
        
        # Show Progress
        if (i + 1) % 50 == 0:
            sample_images()
            print('[Epoch %d/%d] [Batch %d/%d] [D loss : %f] [G loss : %f - (adv : %f, cycle : %f, identity : %f)]'
                  % (epoch + 1, n_epochs, i + 1, len(dataloader), loss_D.item(), loss_G.item(), loss_GAN.item(), loss_cycle.item(), loss_identity.item()))
