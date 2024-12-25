import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm3d(in_channel)
        self.style = nn.Linear(style_dim, in_channel * 2)

    def forward(self, x, style):
        # Generate gamma and beta from style
        style = self.style(style).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        gamma, beta = style.chunk(2, 1)
        # Normalize and apply gamma and beta
        out = self.norm(x)
        return gamma * out + beta


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=512, style_dim=512, num_classes=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim + num_classes, latent_dim),
            nn.LeakyReLU(0.2),
            *[nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LeakyReLU(0.2)
            ) for _ in range(7)]
        )
        self.to_style = nn.Linear(latent_dim, style_dim)

    def forward(self, z, label):
        label_onehot = torch.zeros(z.size(0), 2, device=z.device)
        label_onehot.scatter_(1, label.unsqueeze(1), 1)
        x = torch.cat([z, label_onehot], dim=1)
        x = self.shared(x)
        return self.to_style(x)


class Generator3D(nn.Module):
    def __init__(self, latent_dim=512, style_dim=512, num_classes=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.num_classes = num_classes

        self.mapping = MappingNetwork(latent_dim, style_dim, num_classes)
        self.const = nn.Parameter(torch.randn(1, 512, 4, 4, 4))

        self.conv1 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1)
        self.ada1 = AdaIN(256, style_dim)
        self.conv2 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.ada2 = AdaIN(128, style_dim)
        self.conv3 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.ada3 = AdaIN(64, style_dim)
        self.conv4 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        self.ada4 = AdaIN(32, style_dim)
        self.conv5 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1)
        self.ada5 = AdaIN(16, style_dim)
        self.conv6 = nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1)

        self.activation = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def add_noise(self, x, noise_level=0.05):
        """Add Gaussian noise to the input tensor."""
        return x + torch.randn_like(x) * noise_level

    def forward(self, z, label, noise_inject=True):
        batch_size = z.size(0)
        w = self.mapping(z, label)
        x = self.const.expand(batch_size, -1, -1, -1, -1)

        # Noise injection at each layer
        x = self.conv1(x)
        x = self.ada1(x, w)
        x = self.activation(x)
        if noise_inject:
            x = self.add_noise(x, noise_level=0.05)

        x = self.conv2(x)
        x = self.ada2(x, w)
        x = self.activation(x)
        if noise_inject:
            x = self.add_noise(x, noise_level=0.05)

        x = self.conv3(x)
        x = self.ada3(x, w)
        x = self.activation(x)
        if noise_inject:
            x = self.add_noise(x, noise_level=0.05)

        x = self.conv4(x)
        x = self.ada4(x, w)
        x = self.activation(x)
        if noise_inject:
            x = self.add_noise(x, noise_level=0.05)

        x = self.conv5(x)
        x = self.ada5(x, w)
        x = self.activation(x)
        if noise_inject:
            x = self.add_noise(x, noise_level=0.05)

        x = self.conv6(x)
        return self.tanh(x)  

def compute_gaussian_parameters(latents):
    """
    Computes the mean and covariance of the latent space for slice relationships.
    """
    mean_vector = np.mean(latents, axis=0)
    cov_matrix = np.cov(np.array(latents).T)
    return mean_vector, cov_matrix


def sample_new_latents(mean, cov, num_samples):
    """
    Samples new latent variables based on the Gaussian distribution.
    """
    return np.random.multivariate_normal(mean, cov, size=num_samples)

import nibabel as nib
from torch.utils.data import Dataset, DataLoader


class GliomaBraTSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        for patient_folder in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient_folder)
            if os.path.isdir(patient_path):
                t1c_path = [file for file in os.listdir(patient_path) if 't1c.nii' in file]
                if t1c_path:
                    self.data.append(os.path.join(patient_path, t1c_path[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        volume = nib.load(file_path).get_fdata()
        
        # Normalize real volumes to [-1, 1]
        volume = (volume - volume.min()) / (volume.max() - volume.min())  
        volume = (volume * 2) - 1 
        
        volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  
        if self.transform:
            volume = self.transform(volume)
        return volume

class Discriminator3D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm3d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm3d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm3d(512), nn.LeakyReLU(0.2, inplace=True)
        )
        self.validity_head = nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=0)  
        self.label_head = nn.Linear(512, num_classes) 

    def forward(self, x):
        features = self.feature_extractor(x)  
        validity = self.validity_head(features).view(x.size(0), -1) 
        features_flattened = torch.mean(features, dim=(2, 3, 4)) 
        predicted_labels = self.label_head(features_flattened)
        return validity, predicted_labels

def discriminator_loss(real_validity, fake_validity, real_images, fake_images, lambda_gp=10):
    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
    alpha = torch.rand((real_images.size(0), 1, 1, 1, 1), device=real_images.device)
    interpolated_images = alpha * real_images + (1 - alpha) * fake_images
    interpolated_images.requires_grad_(True)
    mixed_validity, _ = discriminator(interpolated_images)
    gradients = grad(outputs=mixed_validity, inputs=interpolated_images, grad_outputs=torch.ones_like(mixed_validity),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return d_loss + gradient_penalty

def generator_loss(fake_validity):
    return -torch.mean(fake_validity)

import os
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

transform = Compose([])

# Initialize the dataset and dataloader
root_dir = "/scratch1/nissanth/PKG - BraTS-Africa/BraTS-Africa/95_Glioma"  
dataset = GliomaBraTSDataset(root_dir=root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 512
style_dim = 512
num_classes = 2  

generator = Generator3D(latent_dim=latent_dim, style_dim=style_dim, num_classes=num_classes).to(device)

latent_space = []

# Generate latent representations for the dataset
for real_volumes in dataloader:
    real_volumes = real_volumes.to(device)
    z = torch.randn(real_volumes.size(0), latent_dim).to(device)
    labels = torch.randint(0, 2, (real_volumes.size(0),)).to(device)

    # Store latent vectors
    with torch.no_grad():
        latents = generator.mapping(z, labels).cpu().numpy()
        latent_space.extend(latents)

# Compute Gaussian parameters
latent_mean, latent_cov = compute_gaussian_parameters(latent_space)

import torch.optim as optim

# Hyperparameters
latent_dim = 512
style_dim = 512
num_classes = 2
learning_rate = 0.0002
num_epochs = 500

discriminator = Discriminator3D().to(device)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.00001, betas=(0.5, 0.999))

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator3D(latent_dim, style_dim, num_classes).to(device)
optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.99))
loss_fn = nn.MSELoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-5, betas=(0.5, 0.999))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
from torch.autograd import grad
import matplotlib.pyplot as plt
import os

# StyleGAN Trainer Class for 3D MRI Volumes
class StyleGANTrainer:
    def __init__(self, generator, discriminator, device, results_dir='results'):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        self.g_optimizer = optim.Adam(generator.parameters(), lr=0.00001, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-5, betas=(0.5, 0.999))
        self.fixed_noise = torch.randn(4, self.generator.latent_dim).to(device)  
        self.fixed_labels = torch.randint(0, 2, (4,)).to(device)

    def train_step(self, real_volumes, real_labels):
        batch_size = real_volumes.size(0)
        real_volumes = real_volumes.to(self.device)
        real_labels = real_labels.to(self.device)
    
        # Train Discriminator
        self.d_optimizer.zero_grad()
    
        # Generate fake volumes
        z = torch.randn(batch_size, self.generator.latent_dim).to(self.device)
        fake_labels = torch.randint(0, 2, (batch_size,)).to(self.device)
        fake_volumes = self.generator(z, fake_labels).detach() 

        # Resize fake volumes to match real volumes if necessary
        if real_volumes.size() != fake_volumes.size():
            fake_volumes = F.interpolate(fake_volumes, size=real_volumes.shape[2:], mode='trilinear', align_corners=False)
    
        # Discriminator predictions
        real_validity, _ = self.discriminator(real_volumes)
        fake_validity, _ = self.discriminator(fake_volumes)
    
        # WGAN-GP loss with gradient penalty
        gradient_penalty = self.compute_gradient_penalty(real_volumes, fake_volumes)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gradient_penalty
    
        d_loss.backward()
        self.d_optimizer.step()
    
        # Train Generator
        self.g_optimizer.zero_grad()
    
        # Recompute fake volumes for generator training
        fake_volumes = self.generator(z, fake_labels)
        fake_validity, _ = self.discriminator(fake_volumes)
        g_loss = -torch.mean(fake_validity)
    
        g_loss.backward()
        self.g_optimizer.step()
    
        return d_loss.item(), g_loss.item()


    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN-GP"""
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, 1).to(self.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates, _ = self.discriminator(interpolates)
        fake = torch.ones_like(d_interpolates).to(self.device) 
        gradients = grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


    def generate_samples(self, epoch):
        """Generate and visualize samples from the generator."""
        self.generator.eval()
        with torch.no_grad():
            fake_volumes = self.generator(self.fixed_noise, self.fixed_labels)
            fake_volumes_rescaled = (fake_volumes + 1) / 2

            for i in range(fake_volumes_rescaled.size(0)):
                volume = fake_volumes_rescaled[i, 0].cpu().numpy()
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(volume[64, :, :], cmap='gray')  # Axial
                axes[0].set_title('Axial Slice')
                axes[1].imshow(volume[:, 64, :], cmap='gray')  # Coronal
                axes[1].set_title('Coronal Slice')
                axes[2].imshow(volume[:, :, 64], cmap='gray')  # Sagittal
                axes[2].set_title('Sagittal Slice')
                plt.tight_layout()
                plt.savefig(f"{self.results_dir}/epoch_{epoch}_volume_{i}.png")
                plt.close()
        self.generator.train()

def train(generator, discriminator, dataloader, latent_dim, device, epochs=500):
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.5, 0.999))

    trainer = StyleGANTrainer(generator, discriminator, device)

    for epoch in range(epochs):
        g_losses, d_losses = [], []

        for real_volumes in dataloader:
            real_volumes = real_volumes.to(device)
            real_volumes_resized = F.interpolate(real_volumes, size=(128, 128, 128), mode='trilinear', align_corners=False)

            d_loss, g_loss = trainer.train_step(real_volumes_resized, torch.zeros(real_volumes.size(0), dtype=torch.long).to(device))
            d_losses.append(d_loss)
            g_losses.append(g_loss)

        print(f"Epoch {epoch+1}/{epochs}, D Loss: {np.mean(d_losses):.4f}, G Loss: {np.mean(g_losses):.4f}")

        # Generate samples every 10 epochs
        if epoch % 10 == 0:
            trainer.generate_samples(epoch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 512
style_dim = 512
num_classes = 2

# Initialize generator, discriminator, and dataloader
generator = Generator3D(latent_dim, style_dim, num_classes).to(device)
discriminator = Discriminator3D().to(device)

transform = Compose([])
root_dir = "/scratch1/nissanth/PKG - BraTS-Africa/BraTS-Africa/95_Glioma"
dataset = GliomaBraTSDataset(root_dir, transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Start training
train(generator, discriminator, dataloader, latent_dim, device, epochs=500)
num_samples = 10
sampled_latents = sample_new_latents(latent_mean, latent_cov, num_samples)

sampled_latents = torch.tensor(sampled_latents, dtype=torch.float32).to(device)
labels = torch.randint(0, 2, (num_samples,)).to(device)

# Generate new 3D MRI volumes
with torch.no_grad():
    generated_volumes = generator(sampled_latents, labels)

# Visualize or save generated volumes
generated_volumes = generated_volumes.cpu().numpy()

import matplotlib.pyplot as plt

# Visualize slices from a generated 3D volume
for i, volume in enumerate(generated_volumes):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Axial, coronal, and sagittal slices
    axes[0].imshow(volume[0, :, :, volume.shape[3] // 2], cmap='gray')  # Axial
    axes[0].set_title(f"Volume {i + 1} - Axial")

    axes[1].imshow(volume[0, :, volume.shape[2] // 2, :], cmap='gray')  # Coronal
    axes[1].set_title(f"Volume {i + 1} - Coronal")

    axes[2].imshow(volume[0, volume.shape[1] // 2, :, :], cmap='gray')  # Sagittal
    axes[2].set_title(f"Volume {i + 1} - Sagittal")

    plt.tight_layout()
    plt.show()
