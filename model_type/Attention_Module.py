import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, device='cpu'):
        super(SELayer, self).__init__()
        # 使用平均池化将输入的每个通道压缩为一个数值
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 通过两个全连接层实现特征的压缩和再扩展，中间使用 ReLU 激活函数
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # 将需要初始化的模块添加到列表中
        modules_to_init = [self.avg_pool, self.fc]

        # 遍历列表，对每个模块的参数进行初始化
        for module in modules_to_init:
            for name, tensor in module.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(tensor, mean=0, std=0.0001)

        self.to(device)

    def forward(self, x):
        # input:(batch_size, length, in_channels)
        input = x.permute(0, 2, 1)
        b, c, _ = input.size()
        # 压缩
        y = self.avg_pool(input).view(b, c)
        # 通过全连接层
        y = self.fc(y).view(b, c, 1)
        # 得到(batch_size, 1, channels)
        y = y.permute(0, 2, 1)
        # 重新校准通道
        return x * y.expand_as(x), y

class DRSN(nn.Module):
    """Shrinkage function with input-adaptive threshold for DRSN, incorporating element-wise multiplication with pooled features"""
    def __init__(self, in_channels, device='cpu'):
        super(DRSN, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc1 = nn.Linear(in_channels, in_channels)  # First fully connected layer
        self.bn1 = nn.BatchNorm1d(in_channels)  # Batch normalization
        self.fc2 = nn.Linear(in_channels, in_channels)  # Second fully connected layer
        self.bn2 = nn.BatchNorm1d(in_channels)  # Batch normalization
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation

        # 将需要初始化的模块添加到列表中
        modules_to_init = [self.fc1, self.fc2]

        # 遍历列表，对每个模块的参数进行初始化
        for module in modules_to_init:
            for name, tensor in module.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(tensor, mean=0, std=0.0001)

        self.to(device)

    def forward(self, x):
        # input:(batch_size, length, in_channels)
        x = x.permute(0, 2, 1)
        # Global average pooling
        pooled = self.global_pool(x)  # Shape: (batch_size, in_channels, 1)
        pooled_flat = pooled.squeeze(-1)  # Shape: (batch_size, in_channels)

        # Two-layer fully connected network
        out = F.relu(self.bn1(self.fc1(pooled_flat)))  # Shape: (batch_size, in_channels)
        out = F.relu(self.bn2(self.fc2(out)))  # Shape: (batch_size, in_channels)
        adaptive_threshold = self.sigmoid(out)  # Shape: (batch_size, in_channels)

        # Element-wise multiplication with pooled features to get final adaptive threshold
        final_threshold = adaptive_threshold * pooled_flat  # Shape: (batch_size, in_channels)

        # Expand final_threshold to match input shape and apply it
        final_threshold = final_threshold.unsqueeze(2)  # Shape: (batch_size, in_channels, 1)
        abs_x = torch.abs(x)
        shrunk = torch.max(abs_x - final_threshold, torch.zeros_like(abs_x))

        output = torch.sign(x) * shrunk

        # output:(batch_size, length, in_channels)
        output = output.permute(0, 2, 1)
        final_threshold = final_threshold.permute(0, 2, 1)
        
        return output, final_threshold

class SelfAttention(nn.Module):
    def __init__(self, in_channels, heads, device='cpu'):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.head_dim = in_channels // heads

        assert (
            self.head_dim * heads == in_channels
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # 将需要初始化的模块添加到列表中
        modules_to_init = [self.values, self.keys, self.queries]

        # 遍历列表，对每个模块的参数进行初始化
        for module in modules_to_init:
            for name, tensor in module.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(tensor, mean=0, std=0.0001)

        self.to(device)

    def forward(self, query, keys, values):
        # 输入均为(batch, length, in_channels)
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query and keys and then scale
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, head_dim),
        # keys shape: (N, key_len, heads, head_dim),
        # energy: (N, heads, query_len, key_len)

        # Scale
        energy = energy / (self.in_channels ** (1 / 2))

        attention = torch.softmax(energy, dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, device='cpu'):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        for name, tensor in self.fc.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)

        self.to(device)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, device='cpu'):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

        for name, tensor in self.conv1.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)

        self.to(device)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CA_SA_Module(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7, device='cpu'):
        super(CA_SA_Module, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio, device)
        self.spatial_attention = SpatialAttention(kernel_size, device)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# 使用示例
# embed_size = 256
# heads = 8
# N = 10  # batch size
# value_len, key_len, query_len = 15, 15, 15  # sequence lengths
# values = torch.rand(N, value_len, embed_size)
# keys = torch.rand(N, key_len, embed_size)
# queries = torch.rand(N, query_len, embed_size)

# attention = SelfAttention(embed_size, heads)
# out = attention(values, keys, queries)

# print("Output shape:", out.shape)