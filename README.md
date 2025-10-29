# Anime Face Generation using DCGAN

A Deep Convolutional Generative Adversarial Network (DCGAN) implementation for generating anime-style faces. This project trains a GAN to create realistic anime character faces from random noise.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results](#results)
- [Approach & Challenges](#approach--challenges)
- [File Structure](#file-structure)
- [License](#license)

## 🎯 Overview

This project implements a DCGAN (Deep Convolutional Generative Adversarial Network) to generate anime faces. GANs consist of two neural networks - a Generator and a Discriminator - that compete against each other to produce realistic synthetic images.

**Why DCGAN?**

- More memory efficient than StyleGAN2
- Easier to train with stable convergence
- Excellent quality for 64x64 image generation
- Well-documented architecture with proven results

## ✨ Features

- **DCGAN Architecture**: Implements the proven DCGAN design with transposed convolutions
- **Automatic Dataset Download**: Uses kagglehub to download the anime face dataset
- **Training Monitoring**: Real-time progress bars with loss tracking
- **Sample Generation**: Periodic sample generation during training
- **Checkpointing**: Automatic model saving at regular intervals
- **GPU Acceleration**: Full CUDA support for faster training
- **Label Smoothing**: One-sided label smoothing for improved training stability
- **Visualization**: Loss plotting and sample image generation

## 📦 Requirements

```
python
torch
torchvision
numpy
matplotlib
Pillow
tqdm
kagglehub
```

## 🚀 Installation

1. **Clone or download the repository**

2. **Install dependencies**:

```bash
pip install torch torchvision numpy matplotlib pillow tqdm kagglehub
```

3. **Verify CUDA availability** (optional but recommended):

```python
import torch
print(torch.cuda.is_available())  # Should print True if GPU is available
```

## 📊 Dataset

The project uses the **Anime Face Dataset** from Kaggle, which contains thousands of anime character face images.

- **Source**: [splcher/animefacedataset](https://www.kaggle.com/datasets/splcher/animefacedataset)
- **Download**: Automatic via kagglehub
- **Size**: ~63,000 anime face images
- **Format**: RGB images (various sizes, resized to 64x64)

The dataset is automatically downloaded when you run the script. No manual download required!

## 🏗️ Architecture

### Generator Network

The Generator transforms random noise into realistic anime faces using transposed convolutions:

```
Input: Random noise vector (100-dimensional)
    ↓
ConvTranspose2d(100 → 512) + BatchNorm + ReLU  [4x4]
    ↓
ConvTranspose2d(512 → 256) + BatchNorm + ReLU  [8x8]
    ↓
ConvTranspose2d(256 → 128) + BatchNorm + ReLU  [16x16]
    ↓
ConvTranspose2d(128 → 64) + BatchNorm + ReLU   [32x32]
    ↓
ConvTranspose2d(64 → 3) + Tanh                 [64x64]
    ↓
Output: RGB Image (3 channels, 64x64 pixels)
```

**Key Features**:

- Uses transposed convolutions for upsampling
- Batch normalization for training stability
- ReLU activation in hidden layers
- Tanh activation for output (range: -1 to 1)
- ~3.5M parameters

### Discriminator Network

The Discriminator classifies images as real or fake using standard convolutions:

```
Input: RGB Image (3 channels, 64x64 pixels)
    ↓
Conv2d(3 → 64) + LeakyReLU                     [32x32]
    ↓
Conv2d(64 → 128) + BatchNorm + LeakyReLU       [16x16]
    ↓
Conv2d(128 → 256) + BatchNorm + LeakyReLU      [8x8]
    ↓
Conv2d(256 → 512) + BatchNorm + LeakyReLU      [4x4]
    ↓
Conv2d(512 → 1) + Sigmoid                      [1x1]
    ↓
Output: Probability (0 = fake, 1 = real)
```

**Key Features**:

- Standard convolutions for downsampling
- Batch normalization (except first layer)
- LeakyReLU with slope 0.2
- Sigmoid activation for binary classification
- ~2.8M parameters

### Training Strategy

1. **Discriminator Training**:

   - Train on real images (label = 0.9, with smoothing)
   - Train on fake images (label = 0.0)
   - Minimize: `BCE(D(real), 0.9) + BCE(D(fake), 0.0)`

2. **Generator Training**:

   - Generate fake images from noise
   - Try to fool discriminator with "real" labels
   - Minimize: `BCE(D(G(noise)), 1.0)`

3. **Optimization**:
   - Adam optimizer (β₁=0.5, β₂=0.999)
   - Learning rate: 0.0002
   - Binary Cross Entropy loss

## 💻 Usage

### Basic Training

Simply run the script:

```bash
python gan_anime_faces.py
```

The script will:

1. Automatically download the dataset
2. Initialize the models
3. Train for 100 epochs
4. Save samples and checkpoints periodically
5. Generate final results

### Training Output

During training, you'll see:

- Real-time progress bars with loss values
- Discriminator scores: `D(x)` for real images, `D(G(z))` for fake images
- Epoch completion time and average losses
- Sample generation notifications

Example output:

```
Epoch 1/100: 100%|████████| 492/492 [02:15<00:00,  3.63it/s]
D_loss: 0.6543 | G_loss: 2.1234 | D(x): 0.7234 | D(G(z)): 0.3456
```

### Loading a Trained Model

```python
# Load the trained generator
generator = Generator(LATENT_DIM, FEATURE_MAP_G, IMAGE_CHANNELS).to(device)
generator.load_state_dict(torch.load('gan_output/generator_final.pt'))
generator.eval()

# Generate new images
with torch.no_grad():
    noise = torch.randn(16, LATENT_DIM, 1, 1).to(device)
    fake_images = generator(noise)
    fake_images = (fake_images + 1) / 2.0  # Denormalize
    save_image(fake_images, 'generated_faces.png', nrow=4)
```

## ⚙️ Configuration

Key hyperparameters in the script:

### Training Parameters

```python
BATCH_SIZE = 128              # Number of images per batch
NUM_EPOCHS = 100              # Total training epochs
LEARNING_RATE_G = 0.0002      # Generator learning rate
LEARNING_RATE_D = 0.0002      # Discriminator learning rate
BETA1 = 0.5                   # Adam optimizer beta1
BETA2 = 0.999                 # Adam optimizer beta2
```

### Model Parameters

```python
LATENT_DIM = 100              # Size of noise vector
IMAGE_SIZE = 64               # Output image size (64x64)
IMAGE_CHANNELS = 3            # RGB channels
FEATURE_MAP_G = 64            # Generator feature maps
FEATURE_MAP_D = 64            # Discriminator feature maps
```

### Training Configuration

```python
DISCRIMINATOR_STEPS = 1       # D updates per G update
LABEL_SMOOTHING = 0.1         # One-sided label smoothing
SAVE_INTERVAL = 5             # Save checkpoint every N epochs
SAMPLE_INTERVAL = 1           # Generate samples every N epochs
```

### Adjusting for Your Hardware

**For Limited GPU Memory**:

```python
BATCH_SIZE = 64               # Reduce batch size
FEATURE_MAP_G = 32            # Reduce feature maps
FEATURE_MAP_D = 32
```

**For Faster Training**:

```python
BATCH_SIZE = 256              # Increase batch size
NUM_EPOCHS = 50               # Reduce epochs
```

## 📈 Results

### Expected Training Progression

- **Epochs 1-10**: Discriminator dominates, generator produces noise
- **Epochs 10-30**: Generator learns basic face structure (eyes, hair)
- **Epochs 30-60**: Faces become more detailed and coherent
- **Epochs 60-100**: Fine details improve, diversity increases

### Sample Outputs

The training generates:

- **Periodic Samples**: `gan_output/samples/epoch_N.png` - 64 generated faces
- **Final Samples**: `gan_output/final_samples.png` - Best quality output
- **Loss Plots**: `gan_output/training_losses.png` - Training dynamics

### Quality Metrics

Good training indicators:

- **D(x)**: Should stay around 0.6-0.8 (discriminator correctly identifies real images)
- **D(G(z))**: Should start near 0.0, gradually increase to ~0.4-0.5
- **Generator Loss**: Should decrease initially, then stabilize
- **Discriminator Loss**: Should stay relatively stable around 0.5-1.0

## 🔧 Approach & Challenges

### Why DCGAN?

After evaluating multiple GAN architectures, I chose DCGAN for this project because:

1. **Memory Efficiency**: DCGAN uses ~6M parameters compared to StyleGAN2's 30M+
2. **Training Stability**: Well-established architecture with proven convergence
3. **Quality-Speed Tradeoff**: Excellent results for 64x64 generation in reasonable time
4. **Resource Constraints**: Easier to train on consumer GPUs
5. **Documentation**: Extensive research and community support

### Implementation Challenges

#### 1. **Mode Collapse**

- **Problem**: Generator produces limited variety of faces
- **Solution**:
  - Implemented one-sided label smoothing (0.9 instead of 1.0)
  - Used separate optimizer instances for G and D
  - Added dropout consideration for future iterations

#### 2. **Training Instability**

- **Problem**: Loss oscillations and divergence
- **Solution**:
  - Careful weight initialization (N(0, 0.02))
  - Proper normalization ([-1, 1] range)
  - Adam optimizer with β₁=0.5 (as per DCGAN paper)
  - Learning rate scheduling consideration

#### 5. **Convergence Monitoring**

- **Problem**: Difficult to assess training progress
- **Solution**:
  - Real-time progress bars with multiple metrics
  - Periodic sample generation
  - Loss plotting and visualization
  - Checkpoint saving for rollback

### Future Improvements

1. **Architecture Enhancements**:

   - Progressive growing for higher resolutions
   - Self-attention mechanisms
   - Spectral normalization

2. **Quality Improvements**:
   - Higher resolution output (128x128, 256x256)
   - Conditional generation (specific attributes)
   - Style transfer capabilities
