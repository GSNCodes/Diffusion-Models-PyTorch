import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from ddpm import Diffusion
from unet import UNet
from tqdm import tqdm
from utils import *
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def train(args):
    setup_logging(args.exp_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.exp_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        print(f"[INFO] ---- Starting epoch {epoch+1}/{args.epochs}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            predicted_noise = model(x_t, t)
            
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.exp_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.exp_name, f"ckpt.pt"))

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.exp_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 3
    args.image_size = 64
    args.dataset_path = "LandscapeData/"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)

if __name__ == '__main__':
    main()