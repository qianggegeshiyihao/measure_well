#coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


from model_type.GRU import MyGRU
from model_type.LSTM import MyLSTM
from model_type.CNN_BiLSTM import CNN_BiLSTM
from model_type.UNet_1d import UNet1D
from model_type.MMOE import MMoE

# gpu检测
def use_device():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Using GPU')
    else:
        print('No GPU available, use CPU')
    return device


# 训练用样本分成训练集合与验证集
def train_val_sample_split(dataset, train_data_ratio, batch_size):
    train_size = int(train_data_ratio * len(dataset))
    # val_size = len(dataset) - train_size
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return train_dataloader, val_dataloader

# 模型选择与初始化，并输入list中
def model_choose(model_type, device, train, input_size, seq_len, output_size):
    models = []
    # hyper = []
    for mod in model_type:
        if not train:
            mod = mod[:-2]
        if mod == 'LSTM':
            # hyper = Hyper_LSTM()
            model = MyLSTM(input_size=input_size, hidden_size=output_size, 
                           num_layers=3, device=device, bidirectional=False)
            models.append(model)
        elif mod == 'GRU':
            # hyper = Hyper_GRU()
            model = MyGRU(input_size=input_size, hidden_size=output_size, 
                          num_layers=2, device=device, bidirectional=False)
            models.append(model)
        elif mod == 'BiLSTM':
            # hyper = Hyper_LSTM()
            model = MyLSTM(input_size=input_size, hidden_size=output_size, 
                           num_layers=2, device=device, bidirectional=True)
            models.append(model)
        elif mod == 'BiGRU':
            # hyper = Hyper_GRU()
            model = MyGRU(input_size=input_size, hidden_size=output_size, 
                          num_layers=2, device=device, bidirectional=True)
            models.append(model)
        elif mod == 'CNN_BiLSTM':
            model = CNN_BiLSTM(input_channels=input_size, seq_len=seq_len, 
                               output_size=output_size, device=device)
            models.append(model) 
        elif mod == 'MMOE_U2_Net':
            model = MMoE(input_channels=input_size, seq_len=seq_len, task_size=output_size, 
                         hidden_size=1, num_experts=3, task_name='U2_Net', device=device)
            models.append(model) 
        elif mod == 'MMOE_U2_Bridge_Net':
            model = MMoE(input_channels=input_size, seq_len=seq_len, task_size=output_size, 
                         hidden_size=30, num_experts=3, task_name='U2_Bridge_Net', device=device)
            models.append(model) 
        else:
            raise ValueError("Please choose exist model!!")

    return models  

            
# 优化器选择
def optim(optim_name, model, lr, weight_decay):
    if optim_name == "sgd":
        optimizer = torch.optim.SGD(model[0].parameters(), lr=lr)
    elif optim_name == "adam":
        optimizer = torch.optim.Adam(model[0].parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_name == "adagrad":
        optimizer = torch.optim.Adagrad(model[0].parameters(), lr=lr)
    elif optim_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model[0].parameters(), lr=lr)
    else:
        raise NotImplementedError
    return optimizer

# 损失函数选择
def loss_fn(loss_fn_name, y_pred, y_true):
    if loss_fn_name == 'binary_cross_entropy':
        loss = F.binary_cross_entropy(y_pred, y_true)
    elif loss_fn_name == 'mseloss':
        loss = F.mse_loss(y_pred, y_true)
    elif loss_fn_name == 'l1loss':
        loss = F.l1_loss(y_pred, y_true)
    elif loss_fn_name == 'rmseloss':
        loss = torch.sqrt(F.mse_loss(y_pred, y_true))
    else:
        raise NotImplementedError
    return loss

# 对每个模型和图进行存储
def find_available_folder(model_name):
    base_path = 'model'
    model_base_folder = os.path.join(base_path, model_name)
    folder_index = 1

    while True:
        folder_name = f"{model_name}_{folder_index}"
        folder_path = os.path.join(model_base_folder, folder_name)

        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            return folder_name  # 找到未被占用的编号，返回文件夹路径

        folder_index += 1

        
# 模型训练
def model_train(epoches, models, train_dataloader, val_dataloader, device, optimizer, loss_fn_name,
                   model_save, model_type, paint, lr_decay):

    # 学习率自适应衰减
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_decay, patience=30)
    # 定义周期长度T_max，eta_min是学习率下降的最小值，默认为0
    # scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0.00001)

    for k in range(0, len(models)):
        train_loss_list = []
        train_mae_list = []
        train_mse_list = []
        train_rmse_list = []
        val_mae_list = []
        val_rmse_list = []
        val_mse_list = []
        val_r2_list = []
        best_mae = 100
        model_name = find_available_folder(model_type[k])
        folder_dir = os.path.join('model', model_type[k], model_name)
        model_dir = os.path.join(folder_dir, '%s.pt' % model_name)
        for epoch in range(1, epoches + 1):
            models[k].train()
            train_epoch_loss = 0
            train_epoch_mae = 0
            train_epoch_mse = 0
            train_epoch_rmse = 0
            epoch_tic = time.time()

            for train_x, train_y in train_dataloader:

                if (torch.cuda.is_available()):
                    train_x = train_x.to(device)
                    train_y = train_y.type(torch.FloatTensor).to(device)
                optimizer.zero_grad()
                # model = nn.DataParallel(model)
                train_y_pred = models[k](train_x)
                train_loss = loss_fn(loss_fn_name, train_y_pred, train_y)
                train_loss.backward()
                optimizer.step()

                # 计算同一个epoch下在训练集上累积每个批次的loss
                train_epoch_loss += train_loss.item()
                # mean_absolute_error这个函数只能二维，且F里没有mae，所以对每条曲线直接加法
                for ii in range(train_y_pred.shape[-1]):
                    train_mae = mean_absolute_error(train_y.cpu().data.numpy()[:, :, ii], 
                                                    train_y_pred.cpu().data.numpy()[:, :, ii])
                    train_epoch_mae += train_mae
                    # 计算MSE
                    train_mse = mean_squared_error(train_y.cpu().data.numpy()[:, :, ii], 
                                                    train_y_pred.cpu().data.numpy()[:, :, ii])
                    train_epoch_mse += train_mse
                    # 计算RMSE
                    train_rmse = np.sqrt(mean_squared_error(train_y.cpu().data.numpy()[:, :, ii], 
                                                    train_y_pred.cpu().data.numpy()[:, :, ii]))
                    train_epoch_rmse += train_rmse
                
            # 计算平均每个批次的loss和mae以展示,len(train_dataloader)为批次数量
            train_loss_show = train_epoch_loss / len(train_dataloader)
            train_mae_show = train_epoch_mae / len(train_dataloader)
            train_mse_show = train_epoch_mse / len(train_dataloader)
            train_rmse_show = train_epoch_rmse / len(train_dataloader)

            # 每个epoch下每个样本的整条序列的loss
            # train_loss_show = train_epoch_loss / len(train_dataloader.dataset)
            # train_mae_show = train_epoch_mae / len(train_dataloader.dataset)

            if train_loss_show > 50:
                train_loss_show = 50
            if train_mae_show > 50:
                train_mae_show = 50
            if train_mse_show > 50:
                train_mse_show = 50
            if train_rmse_show > 50:
                train_rmse_show = 50
                
            
            train_loss_list.append(train_loss_show)
            train_mae_list.append(train_mae_show)
            train_mse_list.append(train_mse_show)
            train_rmse_list.append(train_rmse_show)

            models[k].eval()
            # 梯度不在下降时发生
            with torch.no_grad():
                val_epoch_mae = 0
                val_epoch_mse = 0
                val_epoch_rmse = 0
                val_epoch_r2 = 0

                for val_x, val_y in val_dataloader:
                    if torch.cuda.is_available():
                        val_x = val_x.to(device)
                    val_y_pred = models[k](val_x).cpu().data.numpy()
                    val_y_true = val_y.cpu().data.numpy()
                
                    for jj in range(val_y_pred.shape[-1]):
                        # 计算MAE
                        val_mae = mean_absolute_error(val_y_true[:, :, jj], 
                                                        val_y_pred[:, :, jj])
                        val_epoch_mae += val_mae
                        # 计算MSE
                        val_mse = mean_squared_error(val_y_true[:, :, jj], val_y_pred[:, :, jj])
                        val_epoch_mse += val_mse
                        # 计算RMSE
                        val_rmse = np.sqrt(mean_squared_error(val_y_true[:, :, jj], 
                                                                val_y_pred[:, :, jj]))
                        val_epoch_rmse += val_rmse
                        # 计算R^2 Score
                        val_r2 = r2_score(val_y_true[:, :, jj], val_y_pred[:, :, jj])
                        val_epoch_r2 += val_r2

                # 整个epoch下每个batch的平均，更好反馈
                val_mae_show = val_epoch_mae / len(val_dataloader)
                val_mse_show = val_epoch_mse / len(val_dataloader)
                val_rmse_show = val_epoch_rmse / len(val_dataloader)
                val_r2_show = val_epoch_r2 / len(val_dataloader)
                # 每个epoch下每个样本的整条序列的指标
                # val_mae_show = val_epoch_mae / len(val_dataloader.dataset)
                # val_rmse_show = val_epoch_rmse / len(val_dataloader.dataset)
                # val_r2_show = val_epoch_r2 / len(val_dataloader.dataset)

                if val_mae_show > 50:
                    val_mae_show = 50
                if val_rmse_show > 50:
                    val_rmse_show = 50
                if val_mse_show > 50:
                    val_mse_show = 50
                    
                # 画图用
                val_mae_list.append(val_mae_show)
                val_rmse_list.append(val_rmse_show)
                val_mse_list.append(val_mse_show)
                val_r2_list.append(val_r2_show)

                # 存下最好的模型
                if val_mae_show < best_mae:
                    best_mae = val_mae_show
                    if model_save:
                        # 检查目录是否存在，如果不存在则创建
                        if not os.path.exists(folder_dir):
                            os.makedirs(folder_dir)
                        torch.save(models[k].state_dict(), model_dir)

                # loss为平均每个批次的
                print('Epoch: %3d        use_time: %.3f' % (epoch, time.time() - epoch_tic))
                print('train_loss: %.8f     train_mae: %.4f     train_mse:%.4f    train_rmse:%.4f'% (
                    train_loss_show, train_mae_show, train_mse_show, train_rmse_show))
                print('Val_Mae: %.4f     val_mse:%.4f   val_rmse: %.4f     val_r2: %.4f' % (
                    val_mae_show, val_mse_show, val_rmse_show, val_r2_show))
            
            # 根据验证集的损失来调整学习率，在每个epoch最后进行
            scheduler.step(val_mse_show)
                
        if paint:
            curve_plt(epoches=epoches, train_loss_list=train_loss_list, 
            val_mae_list=val_mae_list, train_mae_list=train_mae_list,
            val_mse_list=val_mse_list, train_mse_list=train_mse_list,
            val_rmse_list=val_rmse_list, train_rmse_list=train_rmse_list,
            model_name=model_name, folder_dir=folder_dir)
        

# 训练中的loss和acc曲线
def curve_plt(epoches, train_loss_list, val_mae_list, train_mae_list, 
              val_mse_list, train_mse_list,
              val_rmse_list, train_rmse_list,
              model_name, folder_dir):
    EpochList = np.linspace(start=1, stop=epoches, num=epoches)
    fig = plt.figure(figsize=(15, 6))
    fig1 = fig.add_subplot(2, 2, 1)  # 第1个图表位于左上角
    fig2 = fig.add_subplot(2, 2, 2)  # 第2个图表位于右上角
    fig3 = fig.add_subplot(2, 2, 3)  # 第3个图表位于左下角
    fig4 = fig.add_subplot(2, 2, 4)  # 第4个图表位于右下角

    fig1.plot(EpochList, train_loss_list, 'r', label='train_loss')
    fig1.set_title('%s_Loss' % model_name)
    fig1.set_xlabel('Epoches')
    fig1.set_ylabel('LossList')
    fig1.legend()

    fig2.plot(EpochList, train_mae_list, 'r', label='train_mae')
    fig2.plot(EpochList, val_mae_list, 'b', label='val_mae')
    fig2.set_title('%s_mae' % model_name)
    fig2.set_xlabel('Epoches')
    fig2.set_ylabel('MAE')
    fig2.legend()

    fig3.plot(EpochList, train_mse_list, 'r', label='train_mse')
    fig3.plot(EpochList, val_mse_list, 'b', label='val_mse')
    fig3.set_title('%s_mse' % model_name)
    fig3.set_xlabel('Epoches')
    fig3.set_ylabel('MSEList')
    fig3.legend()

    fig4.plot(EpochList, train_rmse_list, 'r', label='train_rmse')
    fig4.plot(EpochList, val_rmse_list, 'b', label='val_rmse')
    fig4.set_title('%s_mse' % model_name)
    fig4.set_xlabel('Epoches')
    fig4.set_ylabel('RMSEList')
    fig4.legend()

    # 调整子图布局参数
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # 调整上下和左右子图之间的间距

    fig.savefig('%s/%s.png'%(folder_dir, model_name))


def model_test(model, model_names, device, test_dataloader):
    for i in range(len(model)):
        model[i].eval()
        model_name = model_names[i][:-2]
        folder_dir = os.path.join('model', model_name, model_names[i])
        model_dir = os.path.join(folder_dir, '%s.pt' % model_names[i])
        with torch.no_grad():
            test_epoch_mae = 0
            test_epoch_mse = 0
            test_epoch_rmse = 0
            test_epoch_r2 = 0
            for test_x, test_y in test_dataloader:
                if torch.cuda.is_available():
                    test_x = test_x.to(device)

                # 载入网络
                if os.path.isfile(model_dir):
                    start = time.time()
                    model[i].load_state_dict(torch.load(model_dir, map_location=device))
                    end = time.time()
                    print("加载模型文件耗时：%.3f" % ((end - start)))
                else:
                    raise NotImplementedError

                test_y_pred = model[i](test_x)
                for ii in range(test_y_pred.shape[-1]):
                    # 计算MAE
                    train_mae = mean_absolute_error(test_y.cpu().data.numpy()[:, :, ii], 
                                                    test_y_pred.cpu().data.numpy()[:, :, ii])
                    test_epoch_mae += train_mae
                    # 计算RMSE
                    test_rmse = np.sqrt(mean_squared_error(test_y.cpu().data.numpy()[:, :, ii], 
                                                           test_y_pred.cpu().data.numpy()[:, :, ii]))
                    test_epoch_rmse += test_rmse
                    # 计算MSE
                    test_mse = mean_squared_error(test_y.cpu().data.numpy()[:, :, ii], 
                                                  test_y_pred.cpu().data.numpy()[:, :, ii])
                    test_epoch_mse += test_mse
                    # 计算R^2 Score
                    test_r2 = r2_score(test_y.cpu().data.numpy()[:, :, ii], 
                                       test_y_pred.cpu().data.numpy()[:, :, ii])
                    test_epoch_r2 += test_r2

            test_mae_show = test_epoch_mae / len(test_dataloader)
            test_mse_show = test_epoch_mse / len(test_dataloader)
            test_rmse_show = test_epoch_rmse / len(test_dataloader)
            test_r2_show = test_epoch_r2 / len(test_dataloader)

        print('Model_Name: %s     test_mae: %.4f     test_mse: %.4f     test_rmse: %.4f    test_r2: %.4f \n' % (
                    model_names[i], test_mae_show, test_mse_show, test_rmse_show, test_r2_show))
    
