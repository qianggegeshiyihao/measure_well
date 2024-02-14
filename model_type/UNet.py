import torch
from torch import nn
from torch.nn import functional as F


# 上/下采样单层模块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, 
                 kernel_size=(3, 3), out_channels=(64, 128), bias=True,
                 upsample=False, device='cpu'):
        super(ConvBlock, self).__init__()

        if len(kernel_size)!=len(out_channels):
            ValueError("Number of kernel_size must be equal to number of num_channels!!")
        
        self.upsample = upsample
        
        self.conv = nn.ModuleList()
        if upsample:
            self.conv.append(nn.ConvTranspose1d(in_channels, in_channels//2, 
                                              kernel_size=2, stride=2))
            
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

    def forward(self, x, skip_input=None):

        if (self.upsample and skip_input is None) or (not self.upsample and skip_input is not None):
            raise ValueError("Up/down is not match to skip_input!!")

        for i, layer in enumerate(self.conv):
            if self.upsample and i==0:
                x = layer(x)
                x = self.concat(x, skip_input)
            else:
                x = layer(x)
        return x
    
    # 跳跃判断
    def concat(self, x1, x2):
        if x1.size()[2] > x2.size()[2]:
            x2 = F.pad(x2, [0, 1])
            output = torch.cat([x1, x2], dim=1)
        elif x1.size()[2] < x2.size()[2]:
            x1 = F.pad(x1, [0, 1])
            output = torch.cat([x1, x2], dim=1)
        else:
            output = torch.cat([x1, x2], dim=1)

        return output

# 下采样模块
class UNet1D_down(nn.Module):
    def __init__(self, in_channels, 
                 device='cpu', conv_kernel=(3, 3)) :
        super(UNet1D_down, self).__init__()
        self.down_conv1 = ConvBlock(in_channels, conv_kernel,
                                out_channels=(64, 64), device=device)
        self.down_conv2 = ConvBlock(64, conv_kernel, 
                                out_channels=(128, 128), device=device)
        self.down_conv3 = ConvBlock(128, conv_kernel,
                                out_channels=(256, 256), device=device)
        
        self.to(device)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        # 下采样部分
        x1 = self.down_conv1(x)

        x2 = F.max_pool1d(x1, kernel_size=2, stride=2)
        x2 = self.down_conv2(x2)

        x3 = F.max_pool1d(x2, kernel_size=2, stride=2)
        x3 = self.down_conv3(x3)

        x_buttom = F.max_pool1d(x3, kernel_size=2, stride=2)
        

        return x_buttom, x1, x2, x3

class UNet1D_up(nn.Module):
    def __init__(self, 
                 device='cpu', conv_kernel=(3, 3)) :
        super(UNet1D_up, self).__init__()
        self.up1_conv = ConvBlock(512, conv_kernel, 
                                  (256, 256), upsample=True, device=device)
      
        self.up2_conv = ConvBlock(256, conv_kernel, 
                                  (128, 128), upsample=True, device=device)
        
        self.up3_conv = ConvBlock(128, conv_kernel, 
                                  (64, 64), upsample=True, device=device)
        
        self.to(device)

    def forward(self, x_buttom, x1, x2, x3):
        # 上采样和跳跃连接
        x5 = self.up1_conv(x_buttom, x3)
        
        x6 = self.up2_conv(x5, x2)  
        
        x7 = self.up3_conv(x6, x1)

        # output = self.final(x7)
        output = x7

        # 输出层
        output = output.permute(0, 2, 1)
        return output

# 合并+跳跃连接
class UNet1D(nn.Module):
    def __init__(self, input_channel, 
                 device='cpu'
                 ):
        super(UNet1D, self).__init__()
        
        self.down = UNet1D_down(input_channel, device=device)
        self.up = UNet1D_up()
        self.buttom_conv = nn.Sequential(
            nn.Conv1d(8, 16, 3, padding=1),
            nn.Conv1d(16, 16, 3, padding=1)
        )
        for name, tensor in self.buttom_conv.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)

        self.to(device)

    def forward(self, x):

        x_buttom, x1, x2, x3 = self.down(x)
        x_buttom = self.buttom_conv(x_buttom)
        output = self.up(x_buttom, x1, x2, x3)

        return output

# Example usage
if __name__ == '__main__':
    batch_size = 10
    length = 30
    input_size = 4
    unet = UNet1D(input_channel=input_size)
    x = torch.randn(batch_size, length, input_size) 
    print(unet(x).shape)
