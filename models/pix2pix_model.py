import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .models import BaseModel
from . import networks

class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    def initialize(self, opt):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # Define Tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize).to(self.device)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize).to(self.device)

        # Load / Define networks.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                        opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids).to(self.device)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                            opt.which_model_netD, opt.n_layers, use_sigmoid, self.gpu_ids).to(self.device)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # Define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # Initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            print('------------ Networks initialized ------------')
            networks.print_network(self.netG)
            networks.print_network(self.netD)
            print('----------------------------------------------')


    def set_input(self, input):
        """
        Set the order of the inputs(A, B) according to the direction.
        After preprocessing the input, we will define real_A, fake_B, real_B according to the training or testing phase.
        """
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']

        self.input_A.resize_(input_A.size()).copy_(input_A).to(self.device)
        self.input_B.resize_(input_B.size()).copy_(input_B).to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A).to(self.device)
        self.fake_B = self.netG.forward(self.real_A).to(self.device)
        self.real_B = Variable(self.input_B).to(self.device)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True).to(self.device) # What is Volatile ?
        self.fake_B = self.netG.forward(self.real_A).to(self.device)
        self.real_B = Variable(self.input_B, volatile=True).to(self.device)

    def get_image_paths(self):
        return self.image_paths


    # Backpropagation functions

    def backward_D(self):
        """For fake image, stop backprop to the Generator by detaching fake_B"""

        # check dimension of inputs
        # print('backward_D step ==> input real A size: {} \tinput real B size: {}\tinput fake B size: {}'.format(
        # self.real_A.size(), self.real_B.size(), self.fake_B.size()))

        # for fake image
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1)).to(self.device)
        self.pred_fake = self.netD.forward(fake_AB.detach()).to(self.device) # Does this only detach the fake_B:?
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # for real image
        real_AB = torch.cat((self.real_A, self.real_B), 1).to(self.device)#.detach() # <- What does mean?
        self.pred_real = self.netD.forward(real_AB).to(self.device)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()


    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1).to(self.device)
        pred_fake = self.netD.forward(fake_AB).to(self.device)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    # Optimizing function

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    # tools to see current states

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                                ('G_L1', self.loss_G_L1.data[0]),
                                ('D_real', self.loss_D_real.data[0]),
                                ('D_fake', self.loss_D_fake.data[0])])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])


    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print("update learning rate: {} -> {}".format(self.old_lr, lr))
        self.old_lr = lr
