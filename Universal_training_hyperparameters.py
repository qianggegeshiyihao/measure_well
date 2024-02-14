#coding=utf-8
# 输入参数
# 特征选择
Feature = ['AC', 'CNL', 'GR', 'SP', 'DEN']
# 目标特征选择(POR,SH,SAND,POR+SAND)
Target = ['SH', 'POR']
# 模型选择(目前可选：GRU、LSTM、BiGRU、BiLSTM、TCN、CNN_BiLSTM、MMOE_U2_Net、MMOE_U2_Bridge_Net)
Model_train_type = ['MMOE_U2_Bridge_Net']
# 选择test模型
Model_test_type = ['MMOE_U2_Bridge_Net_3']
# 输入选择
Train_data = 'sample/train_data.csv'
# 测试选择
Test_data = 'sample/test_data.csv'
# 是否训练
Train = False
# Train = True


# 模型/训练/测试超参
# 每批次样本数量
batch_size = 8
# 训练集比例
train_data_ratio = 0.9
# 每种模型类别跑多少个
num_model = 1
# 单轮循环所有样本回数
epoches = 400
# 是否保存训练模型(一般都true)
model_save = True
# 训练是否加载模型(一般都false)
model_load = False
# 选择binary_cross_entropy(二分类为logloss)或者mseloss(平方差)或者l1loss(差值)
loss_fn_name = 'mseloss'
# 选择优化器(梯度下降)法子，sgd或者adam或者adagrad或者rmsprop
optim_name = 'adam'
# 学习步长
lr = 0.006
# 学习率衰减
lr_decay = 0.6
# l2正则系数
weight_decay = 1e-5
# 训练结果是否画图
paint = True
