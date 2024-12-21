import os
import tqdm
from argparse import ArgumentParser
import mnist_loader
from model import DiffusionModel
import lightning as L
from torchvision import transforms
from torchvision.utils import save_image

def main(hparams):
    
    # Get training and test data
    train_loader, test_loader = mnist_loader.get_mnist_dataset(hparams.batch_size)
    img_shape = 28

    if hparams.load_model:
        model = DiffusionModel.load_from_checkpoint(hparams.model_path, in_size=img_shape*img_shape, noising_steps=hparams.noising_steps)

        # Define transformation
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.Grayscale(num_output_channels=3),  # Convert to RGB
            transforms.ToTensor(),
        ])
        # Save real images
        os.makedirs('real_images', exist_ok=True)
        images, _ = next(iter(test_loader))  # Get the first batch
        for i in range(images.size(0)):
            save_image(transform(transforms.ToPILImage()(images[i])), f'real_images/img_{i}.png')
        
        # Generate and save generated images
        os.makedirs('generated_images', exist_ok=True)
        num_images = len(images)

        # Define transformation for generated images
        gen_transform = transforms.Compose([
            transforms.Resize(299),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

        for idx in tqdm.tqdm(range(num_images)):
            generated_img = model.denoise_image().squeeze().cpu().numpy()
            if generated_img is not None:
                print(f"Generated image {idx} successfully.")
            else:
                print(f"Failed to generate image {idx}.")
            img = transforms.ToPILImage()(generated_img)
            img = gen_transform(img)
            save_image(img, f'generated_images/img_{idx}.png')
            print(f"Saved generated image {idx}.")
    else:

        model = DiffusionModel(img_shape*img_shape, noising_steps=hparams.noising_steps)
        trainer = L.Trainer(accelerator=hparams.accelerator, devices=hparams.devices, 
                            max_epochs=hparams.max_epochs, log_every_n_steps=hparams.log_every_n_steps)
        trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='auto')
    parser.add_argument("--devices", default=1)
    parser.add_argument("--max_epochs", default=100)
    parser.add_argument("--log_every_n_steps", default=10)
    # Load existing model
    parser.add_argument("--load_model", default=True)
    parser.add_argument("--model_path", default='saved_models/diffusion_model_epoch_99.ckpt')
    # Total diffusion steps
    parser.add_argument("--noising_steps", default=1000)
    parser.add_argument("--batch_size", default=128)
    args = parser.parse_args()
    main(args)