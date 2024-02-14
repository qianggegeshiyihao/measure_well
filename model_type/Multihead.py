import torch
from torch import nn
# from torch.nn.modules.transformer import _get_clones
import math
from torch.nn.modules.container import ModuleList
import copy
from model_type.DNN import MyDNN

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiHeadClassifier(nn.Module):
    """ MultiHead Network architecture.

        :param d_model: int,the dim of each unit of input,inputs[2].
        :param feature_num: int,the number of features, inputs[1].
        :param nhead: int, the number of heads.
        :param num_layers: int, the number of layers.
        :param dnn_output: tuple,tuple of positive integer, the layer number and units in each layer of DNN
        :param src_seq_length: int,the length of each feature.
        :param pos_encode: bool,use position encode or not
        :param device: str, ``"cpu"`` or ``"cuda:0"``
        :return (batch_size, dnn_output[-1])torch.Tensor, if dnn_output[-1]=1 the Tensor's shape is (batch_size).

    """

    def __init__(self, d_model, feature_num, nhead, num_layers, dnn_output, src_seq_length,
                 use_bias, pos_encode=False, device='cpu'):
        super(MultiHeadClassifier, self).__init__()

        self.num_layers = num_layers
        self.pos_encode = pos_encode
        if pos_encode:
            self.pos_encoder = Position_Encoding(d_model * feature_num, seq_len=src_seq_length)
        self.multi_head = nn.MultiheadAttention(d_model * feature_num, nhead)
        self.layers = _get_clones(self.multi_head, num_layers)
        self.fc = MyDNN(input_size=src_seq_length * d_model * feature_num, dnn_units=dnn_output, bias=use_bias,
                        device=device)

        self.to(device)

    def forward(self, x):
        if self.pos_encode:
            x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        for mod in self.layers:
            x = mod(x.transpose(0, 1), x.transpose(0, 1), x.transpose(0, 1))[0].transpose(0, 1)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], 1)
        x = self.fc(x)
        return x


class Position_Encoding(nn.Module):
    def __init__(self, input_size=1, seq_len=20):
        super(Position_Encoding, self).__init__()
        '''
        构建位置编码pe
        pe公式为：
        PE(pos,2i/2i+1) = sin/cos(pos/10000^{2i/pos})
        '''
        self.embedding_size = input_size
        pe = torch.zeros(seq_len, input_size)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term2 = torch.pow(torch.tensor(10000.0), torch.arange(0, input_size, 2).float() / input_size)
        div_term1 = torch.pow(torch.tensor(10000.0), torch.arange(1, input_size, 2).float() / input_size)
        # 切片方式，即从0开始，两个步长取一个。即奇数和偶数位置赋值不一样。直观来看就是每一句话的
        pe[:, 0::2] = torch.sin(position * div_term2)
        pe[:, 1::2] = torch.cos(position * div_term1)
        # 这里是为了与x的维度保持一致，释放了一个维度
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.embedding_size)
        x = x + self.pe[:x.size(0), :]
        return x