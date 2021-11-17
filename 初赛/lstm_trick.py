'''
Author: RockyHoo
Date: 2021-11-17 21:48:05
LastEditTime: 2021-11-17 22:44:50
LastEditors: Please set LastEditors
Description: lstm模型预测
FilePath: \Demand_predict\lstm_trick.py
'''
import warnings
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
import tqdm
import xgboost as xgb
warnings.filterwarnings("ignore")
from numpy.random import seed
from sklearn.metrics import mean_squared_error
import torch
from torch.autograd import Variable
import torch.nn as nn
seed(1)
#%%
'''读取数据并填充缺失日期'''
demand_data=pd.read_csv("./data/demand_train.csv")
'''生成test_data'''
from_date="2020-12-01"
end_date="2021-02-28"
test_date_index=pd.date_range(from_date,end_date)
test_data=pd.DataFrame()
#%%
demand_data['过账日期'] = pd.to_datetime(demand_data['过账日期'])
demand_data=demand_data.sort_values(by='过账日期', ascending=True)
product_gp=demand_data.sort_values("过账日期").groupby(["物料编码","工厂编码"],as_index=False)
date_index=pd.date_range("2018-01-01","2020-11-30")
date_df=pd.DataFrame()
date_df["过账日期"]=date_index
demand_filldate=pd.DataFrame()
for group in tqdm.tqdm(product_gp.groups):
    temp_df=product_gp.get_group(group)
    fd_names=["过账日期","工厂编码","物料编码","物料品牌","物料类型","物料品类"]
    #  填充空白日期
    merged_df=pd.DataFrame()
    merged_df["过账日期"]=date_df["过账日期"]
    for fd_name in fd_names:
        merged_df[fd_name]=[list(temp_df[fd_name])[0]]*len(merged_df)
    
    demand_filldate=demand_filldate.append(pd.merge(merged_df,temp_df,how="left").fillna(0))
    #  生成test_data
    temp_test_df=pd.DataFrame()
    temp_test_df["过账日期"]=test_date_index
    for fd_name in fd_names:
        temp_test_df[fd_name]=[list(temp_df[fd_name])[0]]*len(temp_test_df)
    test_data=test_data.append(temp_test_df)
    
demand_data=demand_filldate
demand_data=demand_data.fillna(0)
#%%
'''需要滞后的数据规模大小'''
lag_size = (demand_data['过账日期'].max().date() - demand_data['过账日期'].min().date()).days

''''''
daily_sales = demand_data.groupby('过账日期', as_index=False)['需求量'].sum()  #同一天的所有物料
store_daily_sales = demand_data.groupby(['工厂编码', '过账日期'], as_index=False)['需求量'].sum()  #根据工厂和过账日期
item_daily_sales = demand_data.groupby(['物料编码', '过账日期'], as_index=False)['需求量'].sum()   #根据物料编码和过账日期

'''使用shift构造以日为单位的时间序列'''
def series_to_supervised(data, window=1, lag=1, dropnan=True):
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    # Current timestep (t=0)
    cols.append(data)
    names += [('%s(t)' % (col)) for col in data.columns]
    # Target timestep (t=lag)
    cols.append(data.shift(-lag))
    names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
    # Put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def transform_data(data_df):
    temp_df=data_df.copy()
    temp_df_gp=temp_df.sort_values("过账日期").groupby(["物料编码","工厂编码","过账日期"],as_index=False)
    temp_df_gp=temp_df_gp.agg({'需求量':["sum"]})
    temp_df_gp.columns=['物料编码','工厂编码','过账日期','sales'] 
    return temp_df_gp
    
train_demand_data_gp=transform_data(demand_data)
test_demand_data_gp=transform_data(test_data)

window = 29  #设置滑动窗口大小
lag = lag_size
train_series = series_to_supervised(train_demand_data_gp.drop('过账日期', axis=1), window=window, lag=lag)
test_series = series_to_supervised(test_demand_data_gp.drop('过账日期', axis=1), window=window, lag=lag)

#保持每一行内的物料和商店的一致
last_item = '物料编码(t-%d)' % window
last_store = '工厂编码(t-%d)' % window

def build_series(series):
    series = series[(train_series['工厂编码(t)'] == train_series[last_store])]
    series = series[(train_series['物料编码(t)'] == train_series[last_item])]
    return series
    
train_series = build_series(train_series)
test_series = build_series(test_series)


# 只保留需求量
columns_to_drop = [('%s(t+%d)' % (col, lag)) for col in ['物料编码', '工厂编码']]
for i in range(window, 0, -1):
    columns_to_drop += [('%s(t-%d)' % (col, i)) for col in ['物料编码', '工厂编码']]
    
    
def drop_cols(series):
    series.drop(columns_to_drop, axis=1, inplace=True)
    series.drop(['物料编码(t)', '工厂编码(t)'], axis=1, inplace=True)
    
drop_cols(train_series)
drop_cols(test_series)



'''划分训练集和验证集'''
labels_col = 'sales(t+%d)' % lag_size
labels = train_series[labels_col]
train_series = train_series.drop(labels_col, axis=1)

X_train, X_valid, Y_train, Y_valid = train_test_split(train_series, labels.values, test_size=0.4, random_state=0)
print('Train set shape', X_train.shape)
print('Validation set shape', X_valid.shape)

#%%
'''构造lstm模型'''
class lstm_reg(torch.nn.Module):    
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=3):
        super(lstm_reg, self).__init__()
#       input_size输入向量的长度
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers) # rnn
        self.l1 = torch.nn.Linear(hidden_size, hidden_size) # 回归
        torch.nn.init.xavier_normal_(self.l1.weight)
#         self.l1.weight.data.fill_(50)   #初始化权重
        self.l2=torch.nn.ReLU()
        self.l3 = nn.BatchNorm1d(hidden_size)
        self.l4 = nn.Linear(hidden_size,hidden_size)
        self.l5=torch.nn.ReLU()
        self.l6 = nn.Linear(hidden_size,output_size)
        self.l7=torch.nn.ReLU()
        # 初始化权重
        for name, param in self.lstm.named_parameters():
            # Xavier正态分布
            if name.startswith("weight"):
                torch.nn.init.xavier_normal_(param)
            else:
                torch.nn.init.zeros_(param)
                
#     初始化对应层的权重            
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(10)

    #确保每个维度的长度一致 
    def __getitem__(self, idx):
        # data: seq_len * input_size
        data, label, seq_len = self.train_data[idx]
        # pad_data: max_seq_len * input_size
        pad_data = np.zeros(shape=(self.max_seq_len, data.shape[1]))
        pad_data[0:data.shape[0]] = data
        sample = {'data': pad_data, 'label': label, 'seq_len': seq_len}
        return sample

    # 在使用交叉熵时注意模型输出的是各分类的概率
    def forward(self, x):
        x, _ = self.lstm(x) # (seq, batch, hidden)
#         x.apply(x.init_weights)
        x = self.l1(x[:,-1,:])
#         s, b, h = x.shape
#         x = x.view(s*b, h) # 转换成线性层的输入格式
        x = self.l1(x)
        x=self.l2(x)
        x=self.l4(x)
        x=self.l5(x)
        x=self.l6(x)
        x=self.l7(x)
        return x
    
#%%
'''训练数据'''
model=lstm_reg(30,50)
def Train_Model():
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    for idx in  tqdm.tqdm(range(10000)):
        print("Epoch:{}  Lr:{:.2E}".format(idx,optimizer.state_dict()['param_groups'][0]['lr']))
        var_x = Variable(torch.Tensor(np.array(X_train).reshape(-1,1,30)))
        var_y = Variable(torch.tensor(np.array(Y_train),dtype=torch.float).reshape(-1,1))
        print("1:",var_y)
        out = model(var_x)
        print("out",out)
        loss = criterion(out, var_y)
        optimizer.zero_grad()
        loss.backward()
        print("Loss:%.2f"%loss.item())
        optimizer.step()
        try:
            if idx%100==0:
                print('Epoch: %d, Loss: %.2f'%(idx + 1, loss.item()))
        except Exception as e:
            print(repr(e))
            continue

Train_Model()
#%%
test_x = Variable(torch.Tensor(np.array(X_train).reshape(-1,1,30)))
y_test=model.predict(test_series)