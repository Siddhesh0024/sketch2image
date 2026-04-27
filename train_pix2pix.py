import os
import argparse
import torch
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.pix2pix import UNetGenerator, PatchGANDiscriminator, init_weights
from models.losses import Pix2PixLoss
from data.preprocess import get_loaders


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   default="./data/raw/edges2shoes")
    p.add_argument("--save_dir",    default="./checkpoints/pix2pix")
    p.add_argument("--sample_dir",  default="./samples/pix2pix")
    p.add_argument("--img_size",    type=int, default=256)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--n_epochs",    type=int, default=200)
    p.add_argument("--decay_epoch", type=int, default=100)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--lambda_l1",   type=float, default=100.0)
    p.add_argument("--save_every",  type=int, default=5)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


# ── LR scheduler: linear decay ────────────────────────────────────────────────

def linear_decay(epoch, n_epochs, decay_epoch, lr):
    if epoch < decay_epoch:
        return lr
    return lr * (1 - (epoch - decay_epoch) / (n_epochs - decay_epoch))


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir,   exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # Data
    train_loader, val_loader = get_loaders(
        args.data_root, args.img_size, args.batch_size, args.num_workers
    )

    # Models
    G = UNetGenerator().to(device);           init_weights(G)
    D = PatchGANDiscriminator().to(device);   init_weights(D)

    # Optimizers
    opt_G = Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    criterion = Pix2PixLoss(lambda_l1=args.lambda_l1)
    scaler    = GradScaler()

    for epoch in range(1, args.n_epochs + 1):

        # ── Adjust LR ──
        new_lr = linear_decay(epoch, args.n_epochs, args.decay_epoch, args.lr)
        for pg in opt_G.param_groups: pg["lr"] = new_lr
        for pg in opt_D.param_groups: pg["lr"] = new_lr

        G.train(); D.train()
        g_losses, d_losses = [], []

        for sketch, photo in train_loader:
            sketch, photo = sketch.to(device), photo.to(device)

            # ── Train Discriminator ──
            with autocast():
                fake = G(sketch)
                d_real = D(sketch, photo)
                d_fake = D(sketch, fake.detach())
                loss_D = criterion.discriminator_loss(d_real, d_fake)

            opt_D.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)

            # ── Train Generator ──
            with autocast():
                d_fake = D(sketch, fake)
                loss_G, _, _ = criterion.generator_loss(d_fake, fake, photo)

            opt_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()

            g_losses.append(loss_G.item())
            d_losses.append(loss_D.item())

        print(f"Epoch [{epoch}/{args.n_epochs}]  "
              f"G: {sum(g_losses)/len(g_losses):.4f}  "
              f"D: {sum(d_losses)/len(d_losses):.4f}  "
              f"lr: {new_lr:.6f}")

        # ── Save samples ──
        if epoch % args.save_every == 0:
            G.eval()
            with torch.no_grad():
                sketch_val, photo_val = next(iter(val_loader))
                sketch_val = sketch_val.to(device)
                fake_val   = G(sketch_val)
                grid = torch.cat([sketch_val, fake_val, photo_val.to(device)], dim=0)
                save_image(grid * 0.5 + 0.5,
                           f"{args.sample_dir}/epoch_{epoch:04d}.png",
                           nrow=4)

            torch.save({"G": G.state_dict(), "D": D.state_dict(),
                        "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
                        "epoch": epoch},
                       f"{args.save_dir}/ckpt_epoch_{epoch:04d}.pt")


if __name__ == "__main__":
    train()
