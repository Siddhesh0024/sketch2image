import torch
import torch.nn as nn


# ── U-Net blocks ──────────────────────────────────────────────────────────────

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        return self.block(torch.cat([x, skip], dim=1))


# ── Generator (U-Net 256) ─────────────────────────────────────────────────────

class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, features=64):
        super().__init__()
        # Encoder
        self.e1 = DownBlock(in_ch,        features,     normalize=False)
        self.e2 = DownBlock(features,     features * 2)
        self.e3 = DownBlock(features * 2, features * 4)
        self.e4 = DownBlock(features * 4, features * 8)
        self.e5 = DownBlock(features * 8, features * 8)
        self.e6 = DownBlock(features * 8, features * 8)
        self.e7 = DownBlock(features * 8, features * 8)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU(inplace=True)
        )
        # Decoder
        self.d1 = UpBlock(features * 8,      features * 8, dropout=True)
        self.d2 = UpBlock(features * 8 * 2,  features * 8, dropout=True)
        self.d3 = UpBlock(features * 8 * 2,  features * 8, dropout=True)
        self.d4 = UpBlock(features * 8 * 2,  features * 8)
        self.d5 = UpBlock(features * 8 * 2,  features * 4)
        self.d6 = UpBlock(features * 4 * 2,  features * 2)
        self.d7 = UpBlock(features * 2 * 2,  features)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_ch, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        b  = self.bottleneck(e7)
        d1 = self.d1(b,  e7)
        d2 = self.d2(d1, e6)
        d3 = self.d3(d2, e5)
        d4 = self.d4(d3, e4)
        d5 = self.d5(d4, e3)
        d6 = self.d6(d5, e2)
        d7 = self.d7(d6, e1)
        return self.out(torch.cat([d7, e1], dim=1))


# ── Discriminator (PatchGAN 70×70) ────────────────────────────────────────────

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_ch=6, features=64):   # in_ch = sketch + image
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features,     features * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 4, features * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 8, 1, 4, 1, 1),   # patch output
        )

    def forward(self, sketch, image):
        return self.model(torch.cat([sketch, image], dim=1))


# ── Weight init ───────────────────────────────────────────────────────────────

def init_weights(net, mean=0.0, std=0.02):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, mean, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
