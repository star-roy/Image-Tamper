"""
Improved U-Net Model for Image Tampering Detection
Uses pretrained encoder and decoder with skip connections for better accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNetWithResNet(nn.Module):
    """
    U-Net architecture with pretrained ResNet encoder
    Much better feature extraction than simple CNN
    """
    
    def __init__(self, pretrained=True, encoder='resnet34'):
        super(UNetWithResNet, self).__init__()
        
        # Load pretrained ResNet encoder
        if encoder == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            filters = [64, 64, 128, 256, 512]
        elif encoder == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            filters = [64, 64, 128, 256, 512]
        elif encoder == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            filters = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown encoder: {encoder}")
        
        # Encoder (downsampling path)
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        # Center/Bottleneck
        self.center = DoubleConv(filters[4], filters[4])
        
        # Decoder (upsampling path) - 4 levels to match encoder
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(filters[3] + filters[3], filters[3])
        
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(filters[2] + filters[2], filters[2])
        
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(filters[1] + filters[1], filters[1])
        
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(filters[0], filters[0])
        
        # Final upsampling to compensate for initial maxpool  
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Final output layer
        self.final = nn.Conv2d(filters[0], 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder - 4 levels after initial conv+pool
        enc0 = self.encoder0(x)  # 64, H/4, W/4
        enc1 = self.encoder1(enc0)  # 64, H/4, W/4
        enc2 = self.encoder2(enc1)  # 128, H/8, W/8
        enc3 = self.encoder3(enc2)  # 256, H/16, W/16
        enc4 = self.encoder4(enc3)  # 512, H/32, W/32
        
        # Center
        center = self.center(enc4)
        
        # Decoder with skip connections - 4 levels
        dec4 = self.up4(center)  # 256, H/16, W/16
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.up3(dec4)  # 128, H/8, W/8
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.up2(dec3)  # 64, H/4, W/4
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.decoder2(dec2)
        
        # Final upsampling to original resolution
        dec1 = self.up1(dec2)  # 64, H/2, W/2
        dec1 = self.decoder1(dec1)
        
        # Upsample to match input size (compensate for initial maxpool)
        dec1 = self.final_up(dec1)  # 64, H, W
        
        # Final output
        out = self.final(dec1)  # 1, H, W
        out = torch.sigmoid(out)
        
        return out


class AttentionBlock(nn.Module):
    """Attention mechanism for better feature selection"""
    
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """
    U-Net with Attention mechanisms for improved accuracy
    """
    
    def __init__(self, pretrained=True, encoder='resnet34'):
        super(AttentionUNet, self).__init__()
        
        # Load pretrained ResNet encoder
        if encoder == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            filters = [64, 64, 128, 256, 512]
        elif encoder == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            filters = [64, 64, 128, 256, 512]
        else:
            raise ValueError(f"Unknown encoder: {encoder}")
        
        # Encoder
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        # Center
        self.center = DoubleConv(filters[4], filters[4])
        
        # Attention blocks
        self.att4 = AttentionBlock(F_g=filters[4], F_l=filters[3], F_int=filters[3]//2)
        self.att3 = AttentionBlock(F_g=filters[3], F_l=filters[2], F_int=filters[2]//2)
        self.att2 = AttentionBlock(F_g=filters[2], F_l=filters[1], F_int=filters[1]//2)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(filters[3] + filters[3], filters[3])
        
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(filters[2] + filters[2], filters[2])
        
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(filters[1] + filters[1], filters[1])
        
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(filters[0], filters[0])
        
        # Final upsampling to compensate for initial maxpool
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Final output
        self.final = nn.Conv2d(filters[0], 1, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(enc0)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Center
        center = self.center(enc4)
        
        # Decoder with attention
        dec4 = self.up4(center)
        enc3_att = self.att4(g=dec4, x=enc3)
        dec4 = torch.cat([dec4, enc3_att], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.up3(dec4)
        enc2_att = self.att3(g=dec3, x=enc2)
        dec3 = torch.cat([dec3, enc2_att], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.up2(dec3)
        enc1_att = self.att2(g=dec2, x=enc1)
        dec2 = torch.cat([dec2, enc1_att], dim=1)
        dec2 = self.decoder2(dec2)
        
       # Final upsampling
        dec1 = self.up1(dec2)
        dec1 = self.decoder1(dec1)
        
        # Upsample to match input size
        dec1 = self.final_up(dec1)
        
        # Final output
        out = self.final(dec1)
        out = torch.sigmoid(out)
        
        return out


def get_improved_model(model_type='unet_resnet34', pretrained=True):
    """
    Factory function for improved models
    
    Args:
        model_type: Model architecture type
        pretrained: Use pretrained encoder
    
    Returns:
        model: Model instance
    """
    if model_type == 'unet_resnet34':
        return UNetWithResNet(pretrained=pretrained, encoder='resnet34')
    elif model_type == 'unet_resnet18':
        return UNetWithResNet(pretrained=pretrained, encoder='resnet18')
    elif model_type == 'unet_resnet50':
        return UNetWithResNet(pretrained=pretrained, encoder='resnet50')
    elif model_type == 'attention_unet':
        return AttentionUNet(pretrained=pretrained, encoder='resnet34')
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = get_improved_model('unet_resnet34', pretrained=False)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
