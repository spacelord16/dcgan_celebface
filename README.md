# DCGAN CelebA Face Generation

A Deep Convolutional Generative Adversarial Network (DCGAN) implementation for generating realistic human faces using the CelebA dataset.

## Overview

This project implements a DCGAN to generate synthetic face images. The model consists of a Generator network that creates fake images from random noise and a Discriminator network that distinguishes between real and generated images. Through adversarial training, the Generator learns to produce increasingly realistic face images.

## Features

- DCGAN architecture with convolutional layers
- Custom early stopping mechanism for stable GAN training
- Automatic image generation and saving during training
- Configurable hyperparameters
- Training loss visualization
- Support for large-scale face datasets

## Requirements

```
tensorflow>=2.13.0
numpy>=1.24.0
matplotlib>=3.7.0
Pillow>=10.0.0
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd dcgan_celeba
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the CelebA dataset and place it in the `dataset/` directory with the following structure:

```
dataset/
└── img_align_celeba/
    └── img_align_celeba/
        ├── 000001.jpg
        ├── 000002.jpg
        └── ... (202,599 images)
```

## Usage

### Training

Run the training script:

```bash
python train.py
```

The training will:

- Load and preprocess the CelebA dataset
- Train the DCGAN for the specified number of epochs
- Generate sample images after each epoch
- Save generated images to the `generated/` directory
- Create loss history plots

### Configuration

Modify `config.py` to adjust training parameters:

```python
# Data parameters
DATASET_PATH = "dataset/img_align_celeba/img_align_celeba"
IM_SHAPE = (64, 64, 3)
BATCH_SIZE = 128

# Model parameters
LATENT_DIM = 100

# Training parameters
EPOCHS = 20
LEARNING_RATE = 2e-4
BETA_1 = 0.5

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 0.001
```

## Architecture

### Generator

- Input: Random noise vector (100 dimensions)
- Architecture: Dense -> Reshape -> Conv2DTranspose layers
- Output: RGB image (64x64x3)
- Activation: LeakyReLU for hidden layers, tanh for output

### Discriminator

- Input: RGB image (64x64x3)
- Architecture: Conv2D layers -> Flatten -> Dense
- Output: Binary classification (real/fake probability)
- Activation: LeakyReLU for hidden layers, sigmoid for output

## Early Stopping

The implementation includes a custom early stopping mechanism that monitors:

- NaN or infinite loss values
- Generator loss too low (discriminator completely fooled)
- Discriminator loss too low (generator too weak)
- Combined loss plateau detection

## Training Metrics

The model tracks two primary metrics:

- **d_loss**: Discriminator loss - measures how well the discriminator distinguishes real from fake images
- **g_loss**: Generator loss - measures how well the generator fools the discriminator

Healthy training typically shows both losses oscillating around 0.5-0.7.

## Output

During training, the model generates:

- Sample image grids saved to `generated/gen_images_epoch_XXXX.png`
- Training loss history plot saved as `gan_loss_history.png`
- Console output showing loss values and training progress

## File Structure

```
dcgan_celeba/
├── config.py              # Configuration parameters
├── train.py               # Main training script
├── requirements.txt       # Python dependencies
├── data/
│   ├── __init__.py
│   └── data_loader.py     # Dataset loading and preprocessing
├── models/
│   ├── __init__.py
│   ├── generator.py       # Generator network architecture
│   └── discriminator.py   # Discriminator network architecture
├── training/
│   ├── __init__.py
│   ├── gan_model.py       # GAN training logic
│   └── callbacks.py       # Training callbacks and early stopping
├── generated/             # Generated images and results
└── dataset/              # CelebA dataset (not included in repo)
```

## Training Tips

1. Monitor both generator and discriminator losses during training
2. Adjust learning rates if one network dominates the other
3. Use early stopping to prevent overfitting or mode collapse
4. Check generated image quality visually throughout training
5. Ensure balanced training between generator and discriminator

## Results

The model generates 64x64 pixel face images. Training typically shows improvement in image quality over the first 10-20 epochs. Check the `generated/` folder for sample outputs from each epoch.

## License

This project is for educational purposes. Please ensure compliance with CelebA dataset licensing terms.
