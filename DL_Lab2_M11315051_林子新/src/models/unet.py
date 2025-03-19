# Implement your UNet model here

# assert False, "Not implemented yet!"

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) x2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, features = [64, 128, 256, 512]):
        """
        U-Net Model Implementation in PyTorch.
        
        Parameters:
            in_channels (int): Number of input channels (e.g., 3 for RGB images).
            out_channels (int): Number of output channels (e.g., 1 for segmentation mask).
            features (list): Number of feature maps at each level of the encoder.
        """
        super(UNet, self).__init__()

        # Encoder Path (Downsampling)
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder Path (Upsampling)
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size = 2, stride = 2))
            self.decoder.append(DoubleConv(feature * 2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size = 1)

    def forward(self, x):
        skip_connections = []
        
        # Encoder Path
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size = 2, stride = 2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder Path
        skip_connections = skip_connections[::-1]  # Reverse for correct skip connection
        for up, dec, skip in zip(self.upconvs, self.decoder, skip_connections):
            x = up(x)
            if x.shape != skip.shape:  # Handle misalignment due to cropping
                x = F.interpolate(x, size = skip.shape[2:], mode = "bilinear", align_corners = True)
            x = torch.cat((skip, x), dim = 1)  # Concatenate along channel dimension
            x = dec(x)

        # Final output layer
        return self.final_conv(x)

# # 測試 UNet 模型
# if __name__ == '__main__':
#     # 創建一個 UNet 實例，假設輸入為 3 通道圖像，輸出為 1 通道二值分割圖
#     model = UNet(in_channels=3, out_classes=1, bilinear=True)
#     # 創建一個隨機張量模擬輸入 (batch_size, channels, height, width)
#     x = torch.randn(1, 3, 256, 256)
#     output = model(x)
#     print("輸出尺寸：", output.shape)  # 預期尺寸為 (1, 1, 256, 256)
