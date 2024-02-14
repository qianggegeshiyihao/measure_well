import torch
import torch.nn as nn

from model_type.LSTM import MyLSTM
from model_type.DNN import MyDNN


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_channels, seq_len, output_size,
                 kernel_size=(3, 3), num_channels = (64, 128), hold_len=True,
                 max_pooling=(2, 2),
                 hidden_size=64, num_layers=2, bidirectional=True, bias=True,
                 dnn_units=0, dropout_rate=0, use_bn=True, acti_fn_name='relu',
                 device='cpu'):
        super().__init__()

        # 防止前面卷积导致长度变化
        self.final_seq_len = seq_len

        if len(kernel_size)!=len(num_channels):
            ValueError("Number of kernel_size must be equal to number of num_channels!!")

        self.conv_block = nn.ModuleList()
        for i in range(len(kernel_size)):
            if (kernel_size[i] - 1) % 2 == 1:
                ValueError("Every kernel size must be odd number!!")
            # 保持序列长度，但有padding
            pad_size = (kernel_size[i] - 1) // 2 if hold_len else 0 
            # 求每个卷积核下的序列长
            seq_len = seq_len - kernel_size[i] + 1 if not hold_len else seq_len
            # 每次进入卷积时的input_size
            input_channels = input_channels if i==0 else num_channels[i-1]
            self.conv_block.append(nn.Conv1d(in_channels=input_channels, out_channels=num_channels[i],
                                             kernel_size=kernel_size[i], padding=pad_size, bias=bias))
            self.conv_block.append(nn.ReLU(inplace=True))
            if max_pooling[i]!=0:
                self.conv_block.append(nn.MaxPool1d(kernel_size=max_pooling[i], stride=max_pooling[i]))
                seq_len = seq_len // max_pooling[i]
        
        if max_pooling[-1]!=0:
            self.LSTM = MyLSTM(input_size=num_channels[-1], hidden_size=hidden_size, num_layers=num_layers,
                           bidirectional=bidirectional)
        else:
            self.LSTM = MyLSTM(input_size=num_channels[-1], hidden_size=hidden_size, num_layers=num_layers,
                           bidirectional=bidirectional)
        
        if dnn_units == 0:
            dnn_units = (self.final_seq_len*output_size,)
        else:
            dnn_units = list(dnn_units) + [self.final_seq_len*output_size]
        if hold_len:
            if bidirectional:
                self.fc = MyDNN(input_size=hidden_size*2*seq_len, dnn_units=dnn_units, 
                                dropout_rate=dropout_rate, use_bn=use_bn, acti_fn_name=acti_fn_name)
            else:
                self.fc = MyDNN(input_size=hidden_size*seq_len, dnn_units=dnn_units, 
                                dropout_rate=dropout_rate, use_bn=use_bn, acti_fn_name=acti_fn_name)
        else:
            if bidirectional:
                self.fc = MyDNN(input_size=hidden_size*2*seq_len, dnn_units=dnn_units, 
                                dropout_rate=dropout_rate, use_bn=use_bn, acti_fn_name=acti_fn_name)
            else:
                self.fc = MyDNN(input_size=hidden_size*seq_len, dnn_units=dnn_units, 
                                dropout_rate=dropout_rate, use_bn=use_bn, acti_fn_name=acti_fn_name)
                
        for name, tensor in self.conv_block.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)

        self.to(device)


    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError("input need 3 dim!!")
        
        # batch_size * len * input_size -> batch_size * input_size * len
        x = inputs.permute(0, 2, 1)
        for layer in self.conv_block:
            x = layer(x)
        x = x.permute(0, 2, 1)

        x = self.LSTM(x)
        x = self.fc(x)
        # if x.shape[1] % self.final_seq_len != 0:
        #     ValueError("length trans error!!")    
        output = torch.reshape(x,(x.shape[0], self.final_seq_len, -1))

        return output
