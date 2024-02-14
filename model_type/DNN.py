import torch
from torch import nn


class MyDNN(nn.Module):
    """ DNN Network architecture.

        :param input_size: int,the dim of input,inputs[1]*inputs[2].
        :param dropout_rate: float,the rate of dropout.
        :param use_bn: bool,use batch-normalization or not
        :param dnn_units: list,list of positive integer list, the layer number and units in each layer of DNN
        :param init_std: float,to use as the initialize std of weight
        :param bias: bool,use bias or not
        :param device: str, ``"cpu"`` or ``"cuda:0"``
        :param seq_unit: int, the number of units of each seq_feature
        :param seq_len: int, the length of each seq_feature
        :return (batch_size, dnn_units[-1])torch.Tensor, if dnn_units[-1]=1 the Tensor's shape is (batch_size).

    """

    def __init__(self,
                 input_size, dropout_rate=0, use_bn=True, dnn_units=(256, 128, 1),
                 bias=True, acti_fn_name='relu', device='cpu'):
        global i
        super().__init__()

        # 一般直接将dropout_rate设置为0，除非严重过拟合，但用了drop就不要用BN，两者原理上就有冲突
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        self.acti_fn_name = acti_fn_name
        if len(dnn_units) == 0:
            raise ValueError("DNN_units is empty!!")
        self.dnn_units = [input_size] + list(dnn_units)

        self.linear = nn.ModuleList(
            [nn.Linear(self.dnn_units[i], self.dnn_units[i + 1], bias=bias) for i in range(len(self.dnn_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(self.dnn_units[i + 1]) for i in range(len(self.dnn_units) - 1)])

        # 权重正太分布初始化，mean为正态分布均值，std为正太分布标准差(防止初始化权重过大，导致同一模型多次训练方差太大)，一般不动std
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)

        self.to(device)

    def forward(self, inputs):

        if len(inputs.shape) != 3:
            raise ValueError("input need 3 dim!!")

        # 三维输入变成二维
        x = inputs.contiguous().view(inputs.shape[0], inputs.shape[1] * inputs.shape[2])
        for i in range(len(self.linear)):

            fc = self.linear[i](x)

            # 只用于分类用
            if self.dnn_units[i + 1] == 1:
                x = torch.sigmoid(fc).view(fc.shape[0])
                # x = torch.softmax(fc, dim=1).view(fc.shape[0])
            else:
                if self.use_bn:
                    fc = self.bn[i](fc)

                if self.acti_fn_name == 'relu':
                    fc = nn.ReLU()(fc)
                elif self.acti_fn_name == 'sigmoid':
                    fc = nn.Sigmoid()(fc)
                elif self.acti_fn_name == 'tanh':
                    fc = nn.Tanh()(fc)
                else:
                    raise NotImplementedError

                fc = self.dropout(fc)
                x = fc

        return x