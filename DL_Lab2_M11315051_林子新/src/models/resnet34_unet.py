import torch
import torch.nn as nn
import torch.nn.functional as F

#############################################
# 基本模塊：DoubleConv 與 Up (上採樣) 模塊
#############################################

class DoubleConv(nn.Module):
    """連續兩層卷積 + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """上採樣模塊：先上採樣，再與 skip connection 拼接，最後用 DoubleConv 融合"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 使用反卷積上採樣
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        # x1: 解碼器上層特徵，x2: 編碼器相應層的 skip connection 特徵
        x1 = self.up(x1)
        # 若尺寸不匹配，進行補零操作
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # 拼接後融合
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#############################################
# ResNet34 的基本殘差模塊：BasicBlock
#############################################

class BasicBlock(nn.Module):
    expansion = 1  # 基本塊不改變通道數
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        :param inplanes: 輸入通道數
        :param planes: 輸出通道數
        :param stride: 卷積步長，決定是否下採樣
        :param downsample: 是否需要對 shortcut 路徑進行降維處理
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

#############################################
# ResNet34_UNet 模型：結合 ResNet34 編碼器與 UNet 解碼器
#############################################

class ResNet34_UNet(nn.Module):
    def __init__(self, num_classes=1, bilinear=True):
        """
        :param num_classes: 分割任務中目標類別數（例如二分類可設為 1）
        :param bilinear: 是否使用雙線性上採樣
        """
        super(ResNet34_UNet, self).__init__()
        self.bilinear = bilinear
        self.inplanes = 64
        
        # 編碼器部分（參考 ResNet34 結構）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 輸入圖像通道為3
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 3)      # 輸出尺寸: 64通道
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)  # 輸出尺寸: 128通道
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)  # 輸出尺寸: 256通道
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)  # 輸出尺寸: 512通道
        
        # 解碼器部分：利用 Up 模塊進行上採樣與跳躍連接
        # 設計注意：由於編碼器層級不同，skip connection 來自 conv1 (輸出 64通道)、layer1 (64通道)、layer2 (128通道) 和 layer3 (256通道)
        # layer4 為最底層特徵（512通道）
        self.up1 = Up(512 + 256, 256, bilinear)   # 與 layer3 拼接：512 + 256 = 768
        self.up2 = Up(256 + 128, 128, bilinear)   # 與 layer2 拼接：256 + 128 = 384
        self.up3 = Up(128 + 64, 64, bilinear)     # 與 layer1 拼接：128 + 64 = 192
        self.up4 = Up(64 + 64, 64, bilinear)      # 與 conv1 拼接：64 + 64 = 128
        
        # 最終輸出層：使用 1x1 卷積將通道數轉為 num_classes
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        """建立由多個 BasicBlock 組成的層"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 編碼器部分
        x1 = self.relu(self.bn1(self.conv1(x)))    # x1 尺寸約為原圖的一半 (H/2, W/2)
        x2 = self.maxpool(x1)                        # x2 尺寸約為 H/4, W/4
        x3 = self.layer1(x2)                         # layer1 輸出 (H/4, W/4)
        x4 = self.layer2(x3)                         # layer2 輸出 (H/8, W/8)
        x5 = self.layer3(x4)                         # layer3 輸出 (H/16, W/16)
        x6 = self.layer4(x5)                         # layer4 輸出 (H/32, W/32)
        
        # 解碼器部分：逐步上採樣並與對應的 skip connection 拼接
        d1 = self.up1(x6, x5)  # 將 x6 上採樣後與 layer3 的特徵拼接
        d2 = self.up2(d1, x4)  # 與 layer2 拼接
        d3 = self.up3(d2, x3)  # 與 layer1 拼接
        d4 = self.up4(d3, x1)  # 與 conv1 的特徵拼接
        
        out = self.outc(d4)
        # 若希望輸出與原圖尺寸一致，可額外上採樣一次
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out

#############################################
# 測試 ResNet34_UNet 模型
#############################################

if __name__ == '__main__':
    # 假設二分類分割任務，num_classes 設置為 1
    model = ResNet34_UNet(num_classes=1, bilinear=True)
    # 輸入尺寸假設為 (batch_size, 3, 256, 256)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print("輸出尺寸：", output.shape)  # 預期與輸入相同，即 (1, 1, 256, 256)
