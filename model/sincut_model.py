import numpy as np
import torch
import os
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
#import util.util as util


class SinCUTModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        if self.isTrain:
            if opt.lambda_R1 > 0.0:
                self.loss_names += ['D_R1']
            if opt.lambda_identity > 0.0:
                self.loss_names += ['idt']

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if not self.opt.isTrain:
            self.minmax = input['AREA']
            self.img_tensor = input['IMG']
            self.focus_tensor = input['FOCUS']

        self.image_paths = os.path.join(self.opt.dataroot, 'image.jpg')

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        if self.opt.isTrain:
            self.fake = self.netG(self.real)
            self.fake_B = self.fake[:self.real_A.size(0)]
            if self.opt.nce_idt:
                self.idt_B = self.fake[self.real_A.size(0):]
        else:
            miny, maxy, minx, maxx = self.minmax
            print(self.real.shape)
            self.fake = self.netG(self.real)
            self.fake_B = self.fake[:self.real_A.size(0)]
            img_tensor = self.img_tensor
            for i in range(miny, maxy):
                for j in range(minx, maxx):
                    if self.focus_tensor[i, j] == 0:
                        img_tensor[0, i, j] = self.fake_B[i - miny, j - maxx]
                        img_tensor[1, i, j] = img_tensor[0, i, j]
                        img_tensor[2, i, j] = img_tensor[0, i, j]
            self.fake_B = img_tensor

            '''
            self.real_l = self.real[:,:,:1200,:960]
            self.real_r = self.real[:,:,:1200,960:1920]
            self.fake_l = self.netG(self.real_l)
            torch.cuda.empty_cache()
            self.fake_r = self.netG(self.real_r)
            self.fake_l = self.fake_l[:self.real_l.size(0)]
            self.fake_r = self.fake_r[:self.real_r.size(0)]
            self.fake_B = torch.cat((self.fake_l, self.fake_r), 3)
            '''


    def compute_D_loss(self):
        self.real_B.requires_grad_()

        #GAN_loss_D = super().compute_D_loss()
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        GAN_loss_D  = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D_R1 = self.R1_loss(self.pred_real, self.real_B)
        self.loss_D = GAN_loss_D + self.loss_D_R1
        return self.loss_D

    def compute_G_loss(self):
        #CUT_loss_G = super().compute_G_loss()
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        CUT_loss_G = self.loss_G_GAN + loss_NCE_both

        self.loss_idt = torch.nn.functional.l1_loss(self.idt_B, self.real_B) * self.opt.lambda_identity
        return CUT_loss_G + self.loss_idt

    def R1_loss(self, real_pred, real_img):
        grad_real, = torch.autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True, retain_graph=True)
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty * (self.opt.lambda_R1 * 0.5)

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)

        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)   # nce层数, 原batch数量*挑选的特征点, 原channel经过mlp后的特征
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
