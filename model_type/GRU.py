import torch
from torch import nn


class MyGRU(nn.Module):
    """ GRU Network architecture.

        :param input_size: int,the dim of each unit of input,inputs[2].
        :param hidden_size: int,the number of each hidden_unit(outputs)
        :param num_layers: int, the number of GRU layer
        :param device: str, ``"cpu"`` or ``"cuda:0"``
        :param bidirectional: bool, use bidirectional or not
        :return (batch_size, seq_len, output_size)torch.Tensor

    """

    def __init__(self,
                 input_size, hidden_size=1, num_layers=1, device='cpu', bidirectional=False):
        super().__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

        self.h0 = torch.nn.Parameter(torch.zeros(num_layers, hidden_size))

        # 权重正太分布初始化，mean为正态分布均值，std为正太分布标准差(防止初始化权重过大，导致同一模型多次训练方差太大)，一般不动std
        for name, tensor in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)

        self.to(device)

        # self.hyperparameter = ()

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError("input need 3 dim!!")

        h0 = torch.repeat_interleave(self.h0.unsqueeze(dim=1), repeats=inputs.shape[0], dim=1)
        x, _ = self.gru(inputs, h0)
        return x