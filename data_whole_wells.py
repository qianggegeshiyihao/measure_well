import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from Universal_training_hyperparameters import Feature, Target


class MyDataset(Dataset):
    def __init__(self, input_data):
        
        # 读取csv文件
        origin_data = pd.read_csv(input_data, sep=',')
        
        # 选取所有选中数据
        data = []
        all_input = Feature + Target
        for hist in origin_data.groupby('Name'):
            each_data = []
            for feature in all_input:
                each_data.append(np.array(hist[1][feature]).astype('float32'))
            data.append(each_data)

        # x和y各自转成torch.tensor格式(batch_size, seq_len, input_size)
        self.data_x = torch.from_numpy(np.transpose(data, (0, 2, 1))[:,:,:len(Feature)])
        self.data_y = torch.from_numpy(np.transpose(data, (0, 2, 1))[:,:,-len(Target):])

    def __len__(self):
        return len(self.data_x)
    
    def seq_para(self):
        return self.data_x.shape[1], self.data_x.shape[2], self.data_y.shape[2]
    
    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]