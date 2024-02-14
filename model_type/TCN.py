import torch
from torch import nn
from torch.nn.modules.utils import _single

from model_type.DNN import MyDNN


class Hyper_TCN():
    def __init__(self):

        self.output_units = (256, 1)
        self.num_channels = 4
        self.kernel_size = (3, 5, 7)
        self.downsample_size = (7, 5, 3)
        self.num_layers = 3
        self.concat = True
        self.hold_len = False
        self.use_activate = True
        self.dropout = 0
        self.bias = True
        self.use_bn = False

        self.hyper = ('输出通道数(input_size):%d     层数:%d     是否拼接:%s     是否保持长度:%s\n'     
                      '是否激活函数:%s     剪枝系数:%.2f     是否偏置:%s     标准化:%s\n'
                      '输出层:%s     卷积核尺寸:%s     下采样尺寸:%s' % (self.num_channels, self.num_layers, 
                       self.concat, self.hold_len, self.use_activate, self.dropout, self.bias, self.use_bn,
                       str(self.kernel_size), str(self.downsample_size), str(self.output_units)))


class MyConvBlock(nn.Module):
    """ ConvBlock/ConvEmbedding Network architecture.
        :param input_size: int,the dim of each unit of input,inputs[1].
        :param num_channels: int,the number of each kernel_size input's output_channels
        :param kernel_size: tuple, tuple of positive integer list, the height of kernel in conv
        :param down_sample_size: tuple, tuple of positive integer list, the height of kernel in downsample
        :param seq_len: int, number of units of seq feature in each sample
        :param hold_len: boolen, is or not hold the length of sequence
        :param concat: boolen, is or not concat the sequence
        :param use_activate: boolen, is or not use activate function after convolution
        :param bias: bool,use bias or not
        :param device: str, ``"cpu"`` or ``"cuda:0"``
        :return (batch_size, input_size, total_seq_len)torch.Tensor.

    """

    def __init__(self, input_size, num_channels, kernel_size, down_sample_size, seq_len, hold_len=True, concat=False,
                 use_activate=True, bias=True, device='cpu'):
        super().__init__()

        self.concat = concat
        self.use_activate = use_activate
        # 当保持序列长度时才能直接加
        if (not concat) and hold_len:
            TypeError("sum need to hold len!!")

        self.total_seq_len = 0
        self.conv_block = nn.ModuleList()
        k_seq_len = []
        for i in range(len(kernel_size)):
            if (kernel_size[i] - 1) % 2 == 1:
                ValueError("Every kernel size must be odd number!!")
            # 保持序列长度通常用来做嵌入，但有padding
            pad_size = (kernel_size[i] - 1) // 2 if hold_len else 0
            # 求每个卷积核下的序列长
            k_seq_len.append(seq_len - kernel_size[i] + 1 if not hold_len else seq_len)
            self.conv_block.append(nn.Conv1d(in_channels=input_size, out_channels=num_channels,
                                             kernel_size=kernel_size[i], padding=pad_size, bias=bias))

        self.downsample_block = nn.ModuleList()
        # 对于concat的，由于直接拼接序列长度过长，因此需要下采样，这里采用卷积下采样
        if concat:
            if len(down_sample_size) != len(kernel_size):
                ValueError("down sample size must equal to kernel size if use down-sample!!")
            else:
                for i in range(len(down_sample_size)):
                    self.downsample_block.append(nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                                                           kernel_size=down_sample_size[i],
                                                           stride=_single(len(down_sample_size))))
                    # 当下采样后每个序列长度
                    k_seq_len[i] = (k_seq_len[i] - down_sample_size[i]) // len(kernel_size) + 1
        # 最终输出的序列总长
        self.total_seq_len = sum(k_seq_len) if concat else seq_len

        for name, tensor in self.conv_block.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError("input need 3 dim!!")

        # inputs: (batch_size, input_size, sequence_length)
        conv_outputs = 0
        conv_concat = []
        for i in range(len(self.conv_block)):
            y = self.conv_block[i](inputs)
            if self.use_activate:
                y = nn.ReLU()(y)
            if self.concat:
                y = self.downsample_block[i](y)
                conv_concat.append(y)
                conv_outputs = torch.cat(conv_concat, dim=2)
            else:
                conv_outputs += y

        return conv_outputs


class MyMultiLayerTCN(nn.Module):
    """ MultiLayerTCN Network architecture.
        :param input_size: int,the dim of each unit of input,inputs[1].
        :param output_units: tuple, tuple of positive integer list, the layer number and units in each layer of DNN
        :param num_channels: int,the number of each kernel_size input's output_channels
        :param kernel_size: tuple, tuple of positive integer list, the height of kernel in conv
        :param down_sample_size: tuple, tuple of positive integer list, the height of kernel in downsample
        :param seq_len: int, number of units of seq feature in each sample
        :param hold_len: boolen, is or not hold the length of sequence
        :param concat: boolen, is or not concat the sequence
        :param use_activate: boolen, is or not use activate function after convolution
        :param dropout: float,the rate of dropout.
        :param bias: bool,use bias or not
        :param device: str, ``"cpu"`` or ``"cuda:0"``
        :return (batch_size, linear_units[-1])torch.Tensor, if linear_units[-1]=1 the Tensor's shape is (batch_size).
    """

    def __init__(self, input_size, output_units, num_channels, kernel_size, down_sample_size, num_layers, concat,
                 hold_len, use_activate, seq_len, dropout=0, bias=True, use_bn=False, device='cpu', noneSeq_unit=0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.noneSeq_unit = noneSeq_unit
        self.seq_len = seq_len
        self.input_size = input_size

        seq_len_input = seq_len
        self.tcn_layer = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_size if i == 0 else num_channels
            conv_block = MyConvBlock(input_size=in_channels, num_channels=num_channels,
                                                    kernel_size=kernel_size, down_sample_size=down_sample_size,
                                                    seq_len=seq_len_input, hold_len=hold_len, concat=concat,
                                                    use_activate=use_activate, bias=bias, device=device)
            seq_len_input = conv_block.total_seq_len
            self.tcn_layer.append(conv_block)
        self.fc = MyDNN(input_size=seq_len_input * num_channels, dropout_rate=dropout, use_bn=use_bn,
                        dnn_units=output_units, bias=bias, device=device)

    def forward(self, inputs):
        # inputs: (batch_size, sequence_length*input_size + noneSeq_length, 1)
        if len(inputs.shape) != 3:
            raise ValueError("input need 3 dim!!")

        # inputs: (batch_size, sequence_length*input_size, 1)
        if self.noneSeq_unit != 0:
            inputs = inputs[:, :self.seq_len * self.input_size, :]

        # inputs: (batch_size, sequence_length, input_size)
        inputs = torch.reshape(inputs, (inputs.shape[0], self.seq_len, self.input_size))
        # batch_size * len * input_size -> batch_size * input_size * len
        x = inputs.permute(0, 2, 1)
        for layer in self.tcn_layer:
            x = layer(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], 1))
        output = self.fc(x)

        return output
