"""
Fast Lightweight Model for Quick Training
Can train in 15-20 minutes with decent results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastUNet(nn.Module):
    """Lightweight U-Net for fast training - exam ready version"""
    
    def __init__(self):
        super(FastUNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(3, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        
        # Decoder (upsampling)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        
        # Final output
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def conv_block(self, in_channels, out_channels):
        """Simple conv block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # 32, 256, 256
        enc2 = self.enc2(self.pool(enc1))  # 64, 128, 128
        enc3 = self.enc3(self.pool(enc2))  # 128, 64, 64
        enc4 = self.enc4(self.pool(enc3))  # 256, 32, 32
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))  # 512, 16, 16
        
        # Decoder with skip connections
        dec4 = self.up4(bottleneck)  # 256, 32, 32
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)  # 128, 64, 64
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)  # 64, 128, 128
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)  # 32, 256, 256
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Final output
        out = self.final(dec1)
        out = torch.sigmoid(out)
        
        return out


def get_fast_model():
    """Get fast trainable model"""
    return FastUNet()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = get_fast_model()
    print(f"Parameters: {count_parameters(model):,}")
    
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
