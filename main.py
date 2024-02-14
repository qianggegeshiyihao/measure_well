#coding=utf-8

from Universal_training_hyperparameters import *
from data_whole_wells import MyDataset
# from fit_missing_log import MyDataset
from model_exec import *


if __name__ == '__main__':

    device = use_device()

    # 若为训练，每个模型保存pt时自带一个hyper.json，当测试时自动调用
    if Train:
        dataset = MyDataset(input_data=Train_data)
        seq_len, input_size, output_size = dataset.seq_para() 
        # 训练集/验证集分割
        train_dataloader, val_dataloader = train_val_sample_split(dataset=dataset, 
                                                                      train_data_ratio=train_data_ratio,
                                                                      batch_size=batch_size)

        for i in range(0, num_model):
            # 模型选择与初始化
            models = model_choose(model_type=Model_train_type, device=device, train=Train,
                                  input_size=input_size, seq_len=seq_len, output_size=output_size)
            
            optimizer = optim(optim_name=optim_name, model=models, lr=lr, weight_decay=weight_decay)
            
            # 模型训练
            model_train(epoches=epoches, models=models, train_dataloader=train_dataloader, 
                        val_dataloader=val_dataloader, device=device, optimizer=optimizer, 
                        loss_fn_name=loss_fn_name, model_save=model_save, paint=paint, 
                        model_type=Model_train_type, lr_decay=lr_decay)
                
    # 若为测试
    else:
        dataset = MyDataset(input_data=Test_data)
        test_dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
            )
        
        # 文件中模型读取
        seq_len, input_size, output_size = dataset.seq_para() 
        models = model_choose(model_type=Model_test_type, device=device, train=Train,
                              input_size=input_size, seq_len=seq_len, output_size=output_size)
        
        model_test(model=models, model_names=Model_test_type, device=device, test_dataloader=test_dataloader)
        
        print('Ok')
    