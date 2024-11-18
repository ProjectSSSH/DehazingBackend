# import torch
# import torch.nn as nn
# import numpy as np
# import cv2
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np
# import cv2
# import os
# import random
# from PIL import Image
# import matplotlib.pyplot as plt
# from PIL import ImageFilter
# import torchvision.transforms as transforms

# class ResidualBlock(nn.Module):
#     def _____init_____(self, in_channels, out_channels, stride=1, padding=1):
#         super(ResidualBlock, self)._____init_____()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0) if in_channels != out_channels else None

#     def forward(self, x):
#         identity=x
#         if self.skip_connection:
#             identity=self.skip_connection(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += identity  # Add skip connection
#         out = self.relu(out)
#         return out


# class ImprovedDehazeNet(nn.Module):
#     def _____init_____(self, num_residual_blocks=5):
#         super(ImprovedDehazeNet, self)._____init_____()
        
#         # Initial residual block
#         self.initial_block = ResidualBlock(4, 64)
        
#         # List to hold residual blocks
#         self.residual_layers = nn.ModuleList()
#         in_channels = 64
        
#         # Create a specified number of residual blocks
#         for i in range(num_residual_blocks):
#             out_channels = in_channels * 2 if i % 2 == 0 else in_channels
#             self.residual_layers.append(ResidualBlock(in_channels, out_channels))
#             in_channels = out_channels
        
#         # Attention layers
#         self.spatial_attention = SpatialAttention()
#         self.channel_attention = ChannelAttention(in_channels)
        
#         # Deconvolutional layers for upsampling
#         self.deconv1 = nn.ConvTranspose2d(in_channels, 128, kernel_size=3, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
#         self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
        
#         # Dropout for regularization
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x, dcp_features):
#         # Concatenate input with additional features
#         x = torch.cat((x, dcp_features), dim=1)
        
#         # Pass through the initial block
#         x = self.initial_block(x)
        
#         # Pass through all residual blocks
#         for layer in self.residual_layers:
#             x = layer(x)
        
#         # Attention mechanisms
#         x = self.spatial_attention(x)
#         x = self.channel_attention(x)
        
#         # Upsampling layers
#         x = F.relu(self.deconv1(x))
#         x = self.dropout(F.relu(self.deconv2(x + x)))  # Skip connection
#         x = self.deconv3(x + x)  # Skip connection
        
#         return x

# # Define your ResidualBlock, ChannelAttention, and SpatialAttention classes here    
# class SpatialAttention(nn.Module):
#     def _____init_____(self, kernel_size=7):
#         super(SpatialAttention, self).____init____()
#         assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
#         padding = (kernel_size - 1) // 2
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avg_out, max_out], dim=1)
#         out = self.conv(out)
#         return self.sigmoid(out) * x
    
# class ChannelAttention(nn.Module):
#     def ____init____(self, in_channels, reduction=16):
#         super(ChannelAttention, self).___init___()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
        
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out) * x

# def load_model(model_path):
#     model = ImprovedDehazeNet().to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device), strict=False)  # Set `strict=False` if necessary
#     model.eval()  # Set the model to evaluation mode
#     return model


# # Load the model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ImprovedDehazeNet().to(device)
# model.load_state_dict(torch.load(r"C:\Users\krish\OneDrive\Desktop\Examples\dehazing_backend\improved_dehazenet (6).pth", map_location=device))  # Load the saved model weights
# model.eval()  # Set model to evaluation mode
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms

# Corrected ResidualBlock class
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0) if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        if self.skip_connection:
            identity = self.skip_connection(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

# Spatial Attention class
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x

# Channel Attention class
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

# ImprovedDehazeNet model
class ImprovedDehazeNet(nn.Module):
    def __init__(self, num_residual_blocks=5):
        super(ImprovedDehazeNet, self).__init__()
        self.initial_block = ResidualBlock(4, 64)
        self.residual_layers = nn.ModuleList()
        in_channels = 64
        for i in range(num_residual_blocks):
            out_channels = in_channels * 2 if i % 2 == 0 else in_channels
            self.residual_layers.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(in_channels)
        self.deconv1 = nn.ConvTranspose2d(in_channels, 128, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, dcp_features):
        x = torch.cat((x, dcp_features), dim=1)
        x = self.initial_block(x)
        for layer in self.residual_layers:
            x = layer(x)
        x = self.spatial_attention(x)
        x = self.channel_attention(x)
        x = F.relu(self.deconv1(x))
        x = self.dropout(F.relu(self.deconv2(x + x)))
        x = self.deconv3(x + x)
        return x

# Load the model function
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedDehazeNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    return model

# Example usage: Load and test the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"C:\Users\krish\OneDrive\Desktop\Examples\dehazing_backend\improved_dehazenet (6).pth"
model = load_model(model_path)
print("Model loaded and ready for testing.")
