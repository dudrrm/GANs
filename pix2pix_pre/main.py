import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from solver import Solver


def main(config):
    # Device configuration

    if not os.path.exists(config.sample_path):
        os.makedirs(config.sample_path)

    solver = Solver(config)

    if config.mode == 'train':
        solver.train()
    # elif config.mode == 'test':
        # test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--input_nc', type=int, default=3, help='number of input image channel')
    parser.add_argument('--output_nc', type=int, default=3, help='number of output image channel')
    parser.add_argument('--ngf', type=int, default=256, help='number of Generator features in 1st hidden layer')
    parser.add_argument('--ndf', type=int, default=256, help='number of Discriminator features in 1st hidden layer')
    # parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    # parser.add_argument('--c2_dim', type=int, default=8, help='dimention of domain labels (2nd datset)')
    parser.add_argument('--image_size', type=int, default=2048, help='image resolution')
    # parser.add_argument('--dataset', type=str, default='mnist') #,choices=[])
    # parser.add_argument('--g_conv_dim', type=int, default 64, help='number of conv filters in the first layer of G')
    # parser.add_argument('--d_conv_dim', type=int, default 64, help='number of conv filters in the first layer of D')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=4, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=16, help='number of total iterations for training D')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    # parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--sample_path', type=str, default='./sample_dir', help='location where sampled images are saved')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200, help='test model from this step')

    # Miscellaneous
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    config = parser.parse_args()
    print(config)
    main(config)
