import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super().__init__()
        self.loss = nn.MSELoss() if use_lsgan else nn.BCEWithLogitsLoss()

    def __call__(self, pred, is_real):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


class Pix2PixLoss(nn.Module):
    def __init__(self, lambda_l1=100.0):
        super().__init__()
        self.gan_loss = GANLoss()
        self.l1_loss = nn.L1Loss()
        self.lambda_l1 = lambda_l1

    def generator_loss(self, disc_fake, fake_img, real_img):
        g_gan = self.gan_loss(disc_fake, is_real=True)
        g_l1 = self.l1_loss(fake_img, real_img) * self.lambda_l1
        return g_gan + g_l1, g_gan, g_l1

    def discriminator_loss(self, disc_real, disc_fake):
        d_real = self.gan_loss(disc_real, is_real=True)
        d_fake = self.gan_loss(disc_fake, is_real=False)
        return (d_real + d_fake) * 0.5
