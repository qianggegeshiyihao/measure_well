import torch
import torch.nn as nn
import torch.nn.functional as F

from model_type.LSTM import MyLSTM
from model_type.UNet import *
from model_type.Attention_Module import *


# gate前的准备
class GAP(nn.Module):
    def __init__(self, seq_len, num_experts):
        super(GAP, self).__init__()
        self.fc = nn.Linear(seq_len, num_experts)

    def forward(self, input):
        # # input:(batch_size, length, in_channels)
        # input = input.permute(0, 2, 1)
        # # 可不卷积
        # input = nn.Conv1d(input, 1, 3, padding=1)

        # # 实验到底是序列池化还是channel池化
        # input = input.permute(0, 2, 1)

        input = F.avg_pool1d(input, kernel_size=input.size()[2])
        input = torch.flatten(input, 1)
        input = self.fc(input)

        # output:(batch_size, num_experts)
        return input
    

class SEexpert(nn.Module):
    def __init__(self, in_channels, hidden_size, device='cpu'):
        super().__init__()
        self.SE = SELayer(channel=in_channels, device=device)
        self.LSTM = MyLSTM(in_channels, hidden_size=hidden_size, num_layers=2, 
                                           device=device)
        for name, tensor in self.LSTM.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)

        self.to(device)
    
    def forward(self, input):
        x, _ = self.SE(input) 
        return self.LSTM(x)


class U2_Net(nn.Module):
    def __init__(self, hidden_size, task_size, device):
        super().__init__()
        self.UNet = nn.ModuleList([nn.Sequential(
            UNet1D(hidden_size, device),
            UNet1D(2, device)
        )for _ in range(task_size)])

        self.final = nn.ModuleList([nn.Conv1d(2, 1, kernel_size=1)
                                    for _ in range(task_size)])

        # 将需要初始化的模块添加到列表中
        modules_to_init = [self.UNet, self.final]

        # 遍历列表，对每个模块的参数进行初始化
        for module in modules_to_init:
            for name, tensor in module.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(tensor, mean=0, std=0.0001)

                self.to(device)

    def forward(self, input):
        # input:list[(batch_size, length, in_channels),]
        output = []
        for i,_ in enumerate(input):
            x = self.UNet[i](input[i])
            x = x.permute(0, 2, 1)
            x = self.final[i](x)
            output.append(x.permute(0, 2, 1))

        # output:list[(batch_size, length, in_channels),]
        return output
    

class Bridge(nn.Module):
    def __init__(self, in_channels, heads=8, task_size=2, device='cpu'):
        super(Bridge, self).__init__()
        self.task_size = task_size
        self.SE_Filters = nn.ModuleList([
            SELayer(channel=in_channels, device=device)
        for _ in range(task_size)])

        self.DRSN_Filters = nn.ModuleList([
            DRSN(in_channels=in_channels, device=device)
        for _ in range(task_size)])
        
        self.self_att = nn.ModuleList([
            SelfAttention(in_channels, heads, device)
        for _ in range(task_size)])

        self.to(device)

    def forward(self, input):
        # 输入为[(batch, length, in_channels),]的list
        in_seq = torch.cat(input, dim=1)
        out_list = []
        for i in range(self.task_size):
            _, se_weight = self.SE_Filters[i](input[i])
            filter_seq = in_seq * se_weight.expand_as(in_seq)
        
            filter_seq, _ = self.DRSN_Filters[i](filter_seq)
            
            attention_out = self.self_att[i](input[i], filter_seq, filter_seq)

            out_list.append(attention_out)

        # 输出为[(batch, length, in_channels),]的list
        return out_list


class U2_Bridge_Net(nn.Module):
    def __init__(self, hidden_size, task_size, device='cpu'):
        super(U2_Bridge_Net, self).__init__()
        self.down1 = nn.ModuleList([UNet1D_down(in_channels=hidden_size, device=device)
            for _ in range(task_size)])
        self.conv1_1 = nn.ModuleList([nn.Conv1d(256, 512, 3, padding=1)for _ in range(task_size)])
        self.bridge1 = Bridge(in_channels=512, task_size=task_size, device=device)
        self.conv1_2 = nn.ModuleList([nn.Conv1d(512, 512, 3, padding=1)for _ in range(task_size)])
        self.up1 = nn.ModuleList([UNet1D_up(device=device)for _ in range(task_size)])
        self.down2 = nn.ModuleList([UNet1D_down(in_channels=64, device=device)for _ in range(task_size)])
        self.conv2_1 = nn.ModuleList([nn.Conv1d(256, 512, 3, padding=1)for _ in range(task_size)])  
        self.bridge2 = Bridge(in_channels=512, task_size=task_size, device=device)
        self.conv2_2 = nn.ModuleList([nn.Conv1d(512, 512, 3, padding=1)for _ in range(task_size)])
        self.up2 = nn.ModuleList([UNet1D_up(device=device)for _ in range(task_size)])
        
        self.final = nn.ModuleList([nn.Conv1d(64, 1, kernel_size=1)
                                    for _ in range(task_size)])

        for name, tensor in self.final.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=0.0001)

        self.to(device)

    def forward(self, input):
        # input:list[(batch_size, length, in_channels),]

        down1 = [(self.down1[i](input[i])) for i,_ in enumerate(self.down1)]
        conv1_1 = [(self.conv1_1[i](down1[i][0])).permute(0, 2, 1)for i,_ in enumerate(self.conv1_1)]
        bridge1 = self.bridge1(conv1_1)
        conv1_2 = [self.conv1_2[i](bridge1[i].permute(0, 2, 1))for i,_ in enumerate(self.conv1_2)]
        up1 = [self.up1[i](conv1_2[i], down1[i][1], down1[i][2], down1[i][3]) for i,_ in enumerate(self.up1)]
        
        down2 = [(self.down2[i](up1[i])) for i,_ in enumerate(self.down2)]
        conv2_1 = [(self.conv2_1[i](down2[i][0])).permute(0, 2, 1)for i,_ in enumerate(self.conv2_1)]
        bridge2 = self.bridge2(conv2_1)
        conv2_2 = [self.conv2_2[i](bridge2[i].permute(0, 2, 1))for i,_ in enumerate(self.conv2_2)]
        up2 = [(self.up2[i](conv2_2[i], down2[i][1], down2[i][2], down2[i][3])).permute(0, 2, 1) 
               for i,_ in enumerate(self.up2)]

        output = [(self.final[i](up2[i])).permute(0, 2, 1) for i,_ in enumerate(self.final)]
        
        # output:list[(batch_size, length, in_channels),]
        return output


class MMoE(nn.Module):
    def __init__(self, input_channels, seq_len, task_size,
                 hidden_size, num_experts, task_name='U2_Bridge_Net', device='cpu'):
        super(MMoE, self).__init__()
        self.num_experts = num_experts
        self.seq_len = seq_len
        self.experts = nn.ModuleList([SEexpert(input_channels, hidden_size=hidden_size,  
                                           device=device) for _ in range(num_experts)])
        self.gates = nn.ModuleList([GAP(seq_len, num_experts) for _ in range(task_size)])

        # 改动处

        # task为纯U2-Net
        if task_name == 'U2_Net':
            self.task_model = U2_Net(hidden_size, task_size, device)                                            

        # task为U2_Net+bridge
        elif task_name =='U2_Bridge_Net':
            self.task_model = U2_Bridge_Net(hidden_size, task_size, device)   

        # 将需要初始化的模块添加到列表中
        modules_to_init = [self.gates, self.task_model]

        # 遍历列表，对每个模块的参数进行初始化
        for module in modules_to_init:
            for name, tensor in module.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(tensor, mean=0, std=0.0001)

        self.to(device)

    def forward(self, x):
        # Experts
        expert_outputs = [F.relu(expert(x)) for expert in self.experts]

        # Gates
        gate_outputs = [F.softmax(gate(x), dim=1) for gate in self.gates]

        # Weighted sum of expert outputs for each task
        weighted_outputs = []
        for gate_output in gate_outputs:
            weighted_expert_output = torch.stack([gate_output[:, j].unsqueeze(1).expand(-1, self.seq_len).unsqueeze(-1) * expert_outputs[j] 
                                                  for j in range(self.num_experts)], dim=1)
            weighted_outputs.append(torch.sum(weighted_expert_output, dim=1))

        task_outputs = self.task_model(weighted_outputs)
        
        output = torch.cat(task_outputs, dim=2)

        return output
