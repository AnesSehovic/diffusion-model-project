import torch
import torchvision
from PIL import Image


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def generate_noisy_images(x0, t, alpha_hat, device):
    """
    Adds noise to the image x0 at step t.
    x0: original images
    t: time step (integer)
    """
    # x0 is the original image batch.
    # t is a tensor of time steps for each image in the batch.
    # We compute xt, the noisy image at time t.
    # We also return eps, the noise added, which will be used for training.
    alpha_hat = alpha_hat.to(device)
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None] # 
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None] # 
    eps = torch.randn_like(x0, device=device) # 
    xt = sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * eps # 
    return xt, eps

def denoise_image(model, noise_steps, beta, alpha, alpha_hat):
    """
    Denoise an image using the model
    We start with a random noise image xt.
    We iteratively denoise it by reversing the diffusion process.
    At each step, we use the model to predict the noise and update xt.
    Finally, we plot the generated image.
    """
    with torch.no_grad():
        # Start from pure noise
        xt = torch.randn((1, 1, 28, 28))
        
        for t in reversed(range(noise_steps)):
            t_tensor = torch.tensor([t]).long()
            eps_pred = model(xt, t_tensor)
            xt = (xt - beta[t] / torch.sqrt(1 - alpha_hat[t]) * eps_pred) / torch.sqrt(alpha[t])
            # Optionally add some noise except for t = 0
            if t > 0:
                noise = torch.randn_like(xt)
                xt += torch.sqrt(beta[t]) * noise
        return xt
