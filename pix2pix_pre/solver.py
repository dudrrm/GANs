from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import os
import matplotlib.pyplot as plt


class Solver(object):
    def __init__(self, config):
        """Initialize configurations."""

        # Model configurations.
        self.input_nc = config.input_nc
        self.output_nc = config.output_nc
        self.ngf = config.ngf
        self.ndf = config.ndf
        self.image_size = config.image_size
        # self.dataset = config.dataset


        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        # self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.sample_path = config.sample_path

        # Test configuarations
        self.test_iters = config.test_iters

        # Miscellaneous
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Build the Model
        self.build_model()

    def build_model(self):
        """Create a generator and a discriminator"""

        self.G = Generator(self.input_nc, self.output_nc, self.ngf)
        self.D = Discriminator(self.input_nc, self.output_nc, self.ndf)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]"""
        out = (x + 1) / 2
        return out.clamp_(0,1)


    def train(self):

        # check this data-loader
        # trainTransform  = transforms.Compose([transforms.Grayscale(num_output_channels=1),
        #                                         transforms.ToTensor(),
        #                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Cityscapes = torchvision.datasets.Cityscapes(root='/home/nas1_userE/soyoungyang/cityscapes/train/',
        #                                                 split='train', mode='fine',
        #                                                 target_type='color',
                                                        # transform=trainTransform)
                                   # target_transform=transforms.ToTensor(),
                                   # download=True)
        # check a data instance
        # img, smnt = Cityscapes[0]
        # plt.imshow(img.permut(1, 2, 0), smnt(1, 2, 0))
        # plt.show()

        data_loader = torch.utils.data.DataLoader(dataset='/home/nas1_userE/soyoungyang/cityscapes/train',
                                                    batch_size=self.batch_size, shuffle=True)
        total_step = len(data_loader)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch
        strat_iters = 0

        # Start training.
        print('Start training...')
        for epoch in range(self.num_iters):
            for i, (images, _) in enumerate(data_loader):
                # images = images.reshape(self.batch_size, -1).to(self.device)

                # Create the labels which are later used as inputs for the BCE loss
                real_labels = torch.ones(self.batch_size, 1).to(self.device)
                fake_labels = torch.zeros(self.batch_size, 1).to(self.device)

                # ================================================================== #
                #                      Train the discriminator                       #
                # ================================================================== #

                # Compute loss with real images.
                outputs = self.D(images)
                d_loss_real = criterion(outputs, real_labels)
                real_score = outputs

                # Compute loss with fake images
                z = torch.randn(self.batch_size, self.ngf*8).to(self.device)
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                d_loss_fake = criterion(outputs, fake_labels)
                fake_score = outputs

                # Backprop and optimize
                d_loss = d_loss_real + d_loss_fake
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # ================================================================== #
                #                        Train the generator                         #
                # ================================================================== #

                # Compute loss with fake images
                z = torch.randn(self.batch_size, self.ndf*2).to(self.device)
                fake_images = G(x)
                outputs = D(fake_images)

                # Train G to maximize log(D(G(z)))
                g_loss = criterion(outputs, real_labels)

                # Backprop and optimize
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                if (i+1) % 50 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                            .format(epoch, self.num_iters, i+1, total_step, d_loss.item(), g_loss.item(),
                                    real_score.mean().item(), fake_score.mean().item()))
            # Save real images
            if (epoch+1) == 1:
                images = images.reshape(images.size(0), 1, 28, 28)
                save_image(denorm(images), os.path.join(self.sample_path, 'real_images.png'))

            # Savefake images:
            fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
            save_image(denorm(fake_images), os.path.join(self.sample_path, 'fake_images-{}.png'.format(epoch+1)))

        # Save the model checkpoints
        torch.save(G.state_dict(), 'G.ckpt')
        torch.save(D.state_dict(), 'D.ckpt')
