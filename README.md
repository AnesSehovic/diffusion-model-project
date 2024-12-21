# PyTorch Diffusion

This repository contains a PyTorch implementation of a diffusion model for image generation.

## Running the Application

You can run the main Python script without any arguments using the following command:

```bash
python main.py
```

### Running with Arguments

You can also run the script with various arguments to customize the training and evaluation process. Here are some example commands:

```bash
python main.py \
--accelerator gpu \
--devices 2 \
--max_epochs 10 \
--log_every_n_steps 20 \
--load_model True \
--model_path saved_models/diffusion_model_epoch_49.ckpt \
--noising_steps 1000 \
--batch_size 32
```

### Argument description

`--accelerator`: Specifies the type of accelerator to use (e.g., 'cpu', 'gpu', 'tpu'). Default is 'auto'.
`--devices`: Number of devices to use for training (e.g., number of GPUs). Default is 1.
`--max_epochs`: Maximum number of training epochs. Default is 100.
`--log_every_n_steps`: Frequency of logging during training. Default is 10 steps.
`--load_model`: Whether to load an existing model checkpoint. Default is True.
`--model_path`: Path to the model checkpoint to load. Default is 'saved_models/diffusion_model_epoch_99.ckpt'.
`--noising_steps`: Total number of diffusion steps. Default is 1000.
`--batch_size`: Batch size for training and evaluation. Default is 128.

## Generating the FID scores

To calculate the FID (Fréchet Inception Distance) score, you can use the following command:

```bash
python -m pytorch_fid real_images generated_images
```

> **Note**: When you clone this project, the real_images and generated_images directories are already available with sample images. You can use these directories to calculate the FID score directly.

## Example

1. Run the main script to train the model or generate images:

```bash
python main.py
```

2. Calculate the FID score:

```bash
python -m pytorch_fid real_images generated_images
```

This will output the FID score, which is a measure of the quality of the generated images compared to the real images.