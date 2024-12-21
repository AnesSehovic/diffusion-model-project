import torch
import torch.nn as nn
import lightning as L
import math
from modules import *
from utils import *

class DiffusionModel(L.LightningModule):
    def __init__(self, in_size, noising_steps):
        super().__init__()
        self.noising_steps = noising_steps
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.noising_steps, device=self.device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.in_size = in_size

        bilinear = True
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)
        self.sa1 = SAWrapper(256, 7)
        self.sa2 = SAWrapper(256, 3)
        self.sa3 = SAWrapper(128, 7)

    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_size, embed_size)

    def forward(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1) + self.pos_encoding(t, 128, 14)
        x3 = self.down2(x2) + self.pos_encoding(t, 256, 7)
        x3 = self.sa1(x3)
        x4 = self.down3(x3) + self.pos_encoding(t, 256, 3)
        x4 = self.sa2(x4)
        x = self.up1(x4, x3) + self.pos_encoding(t, 128, 7)
        x = self.sa3(x)
        x = self.up2(x, x2) + self.pos_encoding(t, 64, 14)
        x = self.up3(x, x1) + self.pos_encoding(t, 64, 28)
        output = self.outc(x)
        return output

    def get_loss(self, batch, batch_idx):

        # Sample random time step t for each image in the batch
        ts = torch.randint(0, self.noising_steps, (batch[0].size(0),), device=self.device)#.long()

        # Get noisy images noise_imgs and the noise epsilons
        noise_imgs, epsilons = generate_noisy_images(batch[0], ts, alpha_hat=self.alpha_hat, device=self.device)

        # Predict the noise using the model
        eps_pred = self.forward(noise_imgs, ts.unsqueeze(-1).type(torch.float))

        # Compute the loss
        loss = nn.functional.mse_loss(eps_pred, epsilons)
        return loss

    def denoise_image(self):

        with torch.no_grad():
            # Start from pure noise
            xt = torch.randn((1, 1, 28, 28), device=self.device)
            
            for t in reversed(range(self.noising_steps)):
                t_tensor = torch.tensor([t], device=self.device).long()
                eps_pred = self.forward(xt, t_tensor.view(1, 1).repeat(xt.shape[0], 1)) #model(xt, t_tensor)
                xt = (xt - self.beta[t] / torch.sqrt(1 - self.alpha_hat[t]) * eps_pred) / torch.sqrt(self.alpha[t])
                # Optionally add some noise except for t = 0
                if t > 0:
                    noise = torch.randn_like(xt, device=self.device)
                    xt += torch.sqrt(self.beta[t]) * noise
            return xt

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch, batch_idx)
        # logs metrics for each validation_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer
