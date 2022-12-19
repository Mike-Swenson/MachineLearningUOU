"""
Had to fix an issue where the program didn't contain a main function

 	The basis of a GAN is pitting two neural networks against each other in a two player game in which the generator 
	is trained using real data and its goal is to produce data as close to the input as possible.  
	The discriminator is then fed a shuffled mix of output data from the generator and real data. 
	The discriminator tries to decipher which is real and which is artificial data.  
	This competition drives the loss for each system and ideally generates data that is 
	indistinguishable from the original input distribution.
src
https://paperswithcode.com/method/srgan
"""

import numpy as np

import sys

from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

if __name__ == "__main__":

    cuda = torch.cuda.is_available()
    b1 = 0.5
    b2 = 0.999
    batch_size = 4
    channels = 3
    dataset_name = "img_align_celeba"
    decay_epoch = 100
    epoch = 0
    hr_height = 256
    hr_width = 256
    lr = 0.0002
    n_cpu = 9
    n_epochs = 100

    hr_shape = (hr_height, hr_width)

    # Initialize generator and discriminator
    generator = GeneratorResNet()
    discriminator = Discriminator(input_shape=(channels, *hr_shape))
    feature_extractor = FeatureExtractor()

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_content = torch.nn.L1Loss()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        feature_extractor = feature_extractor.cuda()
        criterion_GAN = criterion_GAN.cuda()
        criterion_content = criterion_content.cuda()
    if epoch != 0:
        # Start with an optimized MSE TO avoid bad initialization
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
        discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    dataloader = DataLoader(
        ImageDataset("./images/img_align_celeba", hr_shape=hr_shape),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )

    for epoch in range(epoch, n_epochs):
        for i, imgs in enumerate(dataloader):
            checkpoint = 0
            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(
                Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))),
                requires_grad=False,
            )
            fake = Variable(
                Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))),
                requires_grad=False,
            )

            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Adversarial loss
            loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr)
            loss_content = criterion_content(gen_features, real_features.detach())

            # Total loss
            loss_G = loss_content + 1e-3 * loss_GAN

            loss_G.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            # Loss of real and fake images
            loss_real = criterion_GAN(discriminator(imgs_hr), valid)
            loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)
        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        checkpoint += 1

        loss_D.backward()
        optimizer_D.step()

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
        )

        batches_done = epoch * len(dataloader) + i

    def saveImageExample(imgs_lr, gen_hr):
        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
        imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
        img_grid = torch.cat((imgs_lr, gen_hr), -1)
        print(checkpoint)
        save_image(img_grid, "generated/%d.png" % batches_done, normalize=False)

    def saveModel():
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
        torch.save(
            discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch
        )
