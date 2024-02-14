import torch
from torch import nn
from torch.nn import functional as F

# 下采样
class ConvBlock(nn.Module):
    def __init__(self, in_channels, 
                 kernel_size=(3, 3), out_channels=(64, 128), bias=True,
                 device='cpu' ):
        super(ConvBlock, self).__init__()

        if len(kernel_size)!=len(out_channels):
            ValueError("Number of kernel_size must be equal to number of num_channels!!")
        
        self.conv = nn.ModuleList()
        for i in range(len(kernel_size)):
            if (kernel_size[i] - 1) % 2 == 1:
                ValueError("Every kernel size must be odd number!!")
            # 保持序列长度，但有padding
            pad_size = (kernel_size[i] - 1) // 2
            # 每次进入卷积时的input_size
            input_channels = in_channels if i==0 else out_channels[i-1]
            self.conv.append(nn.Conv1d(input_channels, out_channels[i], 
                                         kernel_size[i], padding=pad_size, bias=bias))
            self.conv.append(nn.BatchNorm1d(out_channels[i]))
            self.conv.append(nn.ReLU(inplace=True))

        for name, tensor in self.conv.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)

        self.to(device)

    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        return x

# 上采样
class Up(nn.Module):
    def __init__(self, in_channel, up_out_channel, device):
        super(Up, self).__init__()
        if up_out_channel:
            self.up = nn.ConvTranspose1d(in_channel, up_out_channel, 
                                              kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', 
                                       align_corners=True)
            
        for name, tensor in self.up.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)

        self.to(device)

    def forward(self, x):
        return self.up(x)

# 合并+跳跃连接
class UNet1D(nn.Module):
    def __init__(self, input_channel, out_channel, 
                 device, conv_kernel=(3, 3)
                 ):
        super(UNet1D, self).__init__()
        self.down_conv1 = ConvBlock(input_channel, conv_kernel,
                                out_channels=(64, 64), device=device)
        self.down_conv2 = ConvBlock(64, conv_kernel, 
                                out_channels=(128, 128), device=device)
        self.down_conv3 = ConvBlock(128, conv_kernel,
                                out_channels=(256, 256), device=device)
        # self.maxpool = nn.MaxPool1d(2, 2)

        self.up1_conv = ConvBlock(256, conv_kernel, 
                                  (512, 512), device)
        self.up1 = Up(512, 256, device=device)
        self.up2_conv = ConvBlock(512, conv_kernel, 
                                  (256, 256), device)
        self.up2 = Up(256, 128, device=device)
        self.up3_conv = ConvBlock(256, conv_kernel, 
                                  (128, 128), device)
        self.up3 = Up(128, 64, device=device)
        self.up4_conv = ConvBlock(128, conv_kernel, 
                                  (64, 64), device)

        self.final = nn.Conv1d(64, out_channel, kernel_size=1)

        for name, tensor in self.final.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)

        self.to(device)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # 下采样部分
        x1 = self.down_conv1(x)

        x2 = F.max_pool1d(x1, kernel_size=2, stride=2)
        x2 = self.down_conv2(x2)

        x3 = F.max_pool1d(x2, kernel_size=2, stride=2)
        x3 = self.down_conv3(x3)

        x4 = F.max_pool1d(x3, kernel_size=2, stride=2)

        # 上采样和跳跃连接
        x4 = self.up1_conv(x4)
        x4 = self.up1(x4)

        x5 = self.concat(x3, x4)  # 跳跃连接
        x5 = self.up2_conv(x5)
        x5 = self.up2(x5)

        x6 = self.concat(x2, x5)  # 跳跃连接
        x6 = self.up3_conv(x6)
        x6 = self.up3(x6)

        x7 = self.concat(x1, x6)  # 跳跃连接
        x7 = self.up4_conv(x7)

        output = x7
        # output = self.final(x7)

        # 输出层
        output = output.permute(0, 2, 1)
        return output

    # 跳跃判断
    def concat(self, x1, x2):
        if x1.size()[2] > x2.size()[2]:
            x2 = F.pad(x2, [0, 1])
            output = torch.cat([x1, x2], dim=1)
        elif x1.size()[2] < x2.size()[2]:
            x1 = F.pad(x2, [0, 1])
            output = torch.cat([x1, x2], dim=1)
        else:
            output = torch.cat([x1, x2], dim=1)

        return output