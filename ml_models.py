#!/usr/bin/env python3
"""
ML Models and Image Processing Module
Supports various denoising architectures with comprehensive metrics
"""

import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image


class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    def __init__(self, channels: int, kernel_size: int, dropout_rate: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class UNetBlock(nn.Module):
    """U-Net style encoder-decoder block"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)


class SimpleAutoencoder(nn.Module):
    """Simple autoencoder for test mode (4 params)"""
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.decoder(x)
        return x


class DeepAutoencoder(nn.Module):
    """Deep autoencoder with residual connections for real mode (8 params)"""
    def __init__(self, kernel_size: int, stride: int, dropout_rate: float):
        super().__init__()
        padding = kernel_size // 2
        
        # Encoder with residual blocks
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        self.res1 = ResidualBlock(32, kernel_size, dropout_rate)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        self.res2 = ResidualBlock(64, kernel_size, dropout_rate)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        # Decoder with skip connections
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size, stride=stride, padding=padding, output_padding=stride-1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size, stride=stride, padding=padding, output_padding=stride-1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size, stride=stride, padding=padding, output_padding=stride-1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e1 = self.res1(e1)
        e2 = self.enc2(e1)
        e2 = self.res2(e2)
        e3 = self.enc3(e2)
        
        # Decoder with skip connections
        d3 = self.dec3(e3)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d2)
        
        return d1


class UNetAutoencoder(nn.Module):
    """U-Net style autoencoder for advanced mode"""
    def __init__(self, kernel_size: int, dropout_rate: float):
        super().__init__()
        
        # Encoder
        self.enc1 = UNetBlock(3, 32, kernel_size)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = UNetBlock(32, 64, kernel_size)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = UNetBlock(64, 128, kernel_size)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = UNetBlock(128, 256, kernel_size)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = UNetBlock(256, 128, kernel_size)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = UNetBlock(128, 64, kernel_size)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = UNetBlock(64, 32, kernel_size)
        
        self.final = nn.Conv2d(32, 3, 1)
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        b = self.dropout(b)
        
        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(d1))


class MLImageProcessor:
    """ML-based image processing with multiple architectures"""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.original = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load CIFAR10 dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        self.test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
    
    def train_simple_model(self, params: Dict) -> Tuple[nn.Module, Dict[str, float]]:
        """Train simple autoencoder (test mode)"""
        start_time = time.time()
        
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=params['batch_size'], 
            shuffle=True, num_workers=2
        )
        
        model = SimpleAutoencoder(params['kernel_size']).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.MSELoss()
        
        # Training
        model.train()
        for epoch in range(params['epochs']):
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                # Add noise
                noise = torch.randn_like(data) * 0.1
                noisy = torch.clamp(data + noise, -1, 1)
                
                optimizer.zero_grad()
                output = model(noisy)
                loss = criterion(output, data)
                loss.backward()
                optimizer.step()
                
                if batch_idx >= 50:  # Limit batches for speed
                    break
        
        # Evaluation
        metrics = self._evaluate_model(model, criterion)
        metrics['runtime'] = time.time() - start_time
        
        return model, metrics
    
    def train_deep_model(self, params: Dict) -> Tuple[nn.Module, Dict[str, float]]:
        """Train deep autoencoder (real mode)"""
        start_time = time.time()
        
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=params['batch_size'], 
            shuffle=True, num_workers=2
        )
        
        model = DeepAutoencoder(
            params['kernel_size'], params['stride'], params['dropout_rate']
        ).to(self.device)
        
        optimizer = optim.SGD(
            model.parameters(), 
            lr=params['lr'], 
            momentum=params['momentum'],
            weight_decay=params['weight_decay']
        )
        criterion = nn.MSELoss()
        
        # Training
        model.train()
        for epoch in range(params['epochs']):
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                noise = torch.randn_like(data) * 0.1
                noisy = torch.clamp(data + noise, -1, 1)
                
                optimizer.zero_grad()
                output = model(noisy)
                loss = criterion(output, data)
                loss.backward()
                optimizer.step()
                
                if batch_idx >= 100:
                    break
        
        # Evaluation
        metrics = self._evaluate_model(model, criterion)
        metrics['runtime'] = time.time() - start_time
        
        return model, metrics
    
    def train_unet_model(self, params: Dict) -> Tuple[nn.Module, Dict[str, float]]:
        """Train U-Net autoencoder (advanced mode)"""
        start_time = time.time()
        
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=params['batch_size'], 
            shuffle=True, num_workers=2
        )
        
        model = UNetAutoencoder(
            params['kernel_size'], params['dropout_rate']
        ).to(self.device)
        
        optimizer = optim.Adam(
            model.parameters(), 
            lr=params['lr'],
            weight_decay=params.get('weight_decay', 0.0001)
        )
        criterion = nn.MSELoss()
        
        # Training
        model.train()
        for epoch in range(params['epochs']):
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                noise = torch.randn_like(data) * 0.15
                noisy = torch.clamp(data + noise, -1, 1)
                
                optimizer.zero_grad()
                output = model(noisy)
                loss = criterion(output, data)
                loss.backward()
                optimizer.step()
                
                if batch_idx >= 150:
                    break
        
        # Evaluation
        metrics = self._evaluate_model(model, criterion)
        metrics['runtime'] = time.time() - start_time
        
        return model, metrics
    
    def _evaluate_model(self, model: nn.Module, criterion: nn.Module) -> Dict[str, float]:
        """Evaluate model on test set"""
        model.eval()
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=32, shuffle=False
        )
        
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        total_mae = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(test_loader):
                data = data.to(self.device)
                noise = torch.randn_like(data) * 0.1
                noisy = torch.clamp(data + noise, -1, 1)
                
                output = model(noisy)
                total_loss += criterion(output, data).item() * data.size(0)
                
                # Convert to numpy for metric calculation
                data_np = data.cpu().numpy()
                output_np = output.cpu().numpy()
                
                for i in range(data.size(0)):
                    img_orig = ((data_np[i].transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
                    img_out = ((output_np[i].transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
                    
                    total_psnr += self._calculate_psnr(img_orig, img_out)
                    total_ssim += self._calculate_ssim(img_orig, img_out)
                    total_mae += np.mean(np.abs(img_orig - img_out))
                    num_samples += 1
                
                if batch_idx >= 50:  # Limit evaluation batches
                    break
        
        return {
            'loss': total_loss / num_samples,
            'psnr': total_psnr / num_samples,
            'ssim': total_ssim / num_samples,
            'mae': total_mae / num_samples
        }
    
    def process_image(self, model: nn.Module, output_path: str = None) -> np.ndarray:
        """Apply trained model to the input image"""
        model.eval()
        
        # Resize and normalize
        img_resized = cv2.resize(self.original, (32, 32)).astype(np.float32)
        img_normalized = (img_resized / 127.5) - 1.0
        
        tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        noise = torch.randn_like(tensor) * 0.1
        noisy = torch.clamp(tensor + noise, -1, 1)
        
        with torch.no_grad():
            output = model(noisy)
        
        # Denormalize and convert back
        output_np = output.cpu().squeeze().numpy().transpose(1, 2, 0)
        output_np = ((output_np + 1) * 127.5).astype(np.uint8)
        
        # Resize to original dimensions
        result = cv2.resize(output_np, (self.original.shape[1], self.original.shape[0]))
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result).save(output_path)
        
        return result
    
    def _calculate_psnr(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
        if mse == 0:
            return 100.0
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))
    
    def _calculate_ssim(self, original: np.ndarray, processed: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        original = original.astype(float)
        processed = processed.astype(float)
        
        mu1 = cv2.GaussianBlur(original, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(processed, (11, 11), 1.5)
        
        sigma1_sq = cv2.GaussianBlur(original ** 2, (11, 11), 1.5) - mu1 ** 2
        sigma2_sq = cv2.GaussianBlur(processed ** 2, (11, 11), 1.5) - mu2 ** 2
        sigma12 = cv2.GaussianBlur(original * processed, (11, 11), 1.5) - mu1 * mu2
        
        ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return float(np.mean(ssim_map))
