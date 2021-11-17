#%%
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
date_sales = demand_data.drop(['工厂编码','物料编码'], axis=1).copy()
#%%
date_sales=date_sales.set_index("过账日期",drop=True)
# 将按月的波动随时间的变化展示出来
ts_diff = date_sales - date_sales.shift(30)
plt.figure(figsize=(22,10))
plt.plot(ts_diff[:20000])
plt.title("Differencing method") 
plt.xlabel("Date")
plt.ylabel("Differencing Sales")
plt.show()
#%%
# 总共的天数
lag_size = (demand_data['过账日期'].max().date() - demand_data['过账日期'].min().date()).days
#%%
daily_sales = demand_data.groupby('过账日期', as_index=False)['需求量'].sum()
store_daily_sales = demand_data.groupby(['工厂编码', '过账日期'], as_index=False)['需求量'].sum()
item_daily_sales = demand_data.groupby(['物料编码', '过账日期'], as_index=False)['需求量'].sum()
#%%
#构造日期特征
import re
def add_datepart(df, fldname, drop=True):

    """
    Parameters:
    -----------
    df: A pandas data frame. df gain several new columns.
    fldname: A string that is the name of the date column you wish to expand.
        If it is not a datetime64 series, it will be converted to one with pd.to_datetime.
    drop: If true then the original date column will be removed.
    """
    
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
        
    targ_pre =""
    
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear','weekofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    
    for n in attr: 
        df[targ_pre + n] = getattr(fld.dt, n.lower())
        
    if drop: 
        df.drop(fldname, axis=1, inplace=True)

add_datepart(demand_data,'过账日期',False)
add_datepart(test_data,'过账日期',False)
#%%
from scipy import stats
original_target = demand_data["需求量"].values
# target为返回的boxcox变换序列
target, lambda_prophet = stats.boxcox(demand_data["需求量"] + 1)
len_train=target.shape[0]
merged_df = pd.concat([demand_data,test_data])
# %%
# 构造统计特征
'''计算特征分组后的需求量的中位值和均值'''
merged_df["median-store_item"] = merged_df.groupby(["物料编码", "工厂编码"])["需求量"].transform("median")
merged_df["mean-store_item"] = merged_df.groupby(["物料编码", "工厂编码"])["需求量"].transform("mean")
merged_df["mean-Month_item"] = merged_df.groupby(["Month", "物料编码"])["需求量"].transform("mean")
merged_df["median-Month_item"] = merged_df.groupby(["Month", "物料编码"])["需求量"].transform("median")
merged_df["median-Month_store"] = merged_df.groupby(["Month", "工厂编码"])["需求量"].transform("median")
merged_df["median-item"] = merged_df.groupby(["物料编码"])["需求量"].transform("median")
merged_df["median-store"] = merged_df.groupby(["工厂编码"])["需求量"].transform("median")
merged_df["mean-item"] = merged_df.groupby(["物料编码"])["需求量"].transform("mean")
merged_df["mean-store"] = merged_df.groupby(["工厂编码"])["需求量"].transform("mean")
merged_df["median-store_item-Month"] = merged_df.groupby(['Month', "物料编码", "工厂编码"])["需求量"].transform("median")
merged_df["mean-store_item-week"] = merged_df.groupby(["物料编码", "工厂编码",'weekofyear'])["需求量"].transform("mean")
merged_df["item-Month-mean"] = merged_df.groupby(['Month', "物料编码"])["需求量"].transform("mean")# mean 需求量 of that item  for all stores scaled
merged_df["store-Month-mean"] = merged_df.groupby(['Month', "工厂编码"])["需求量"].transform("mean")# mean 需求量 of that store  for all items scaled

'''adding more lags (Check the rationale behind this in the links attached)
   制作不同时间间隔步长的时间序列特征
'''
lags = [90,91,98,105,112,119,126,182,189,364]
for lag in lags:
    merged_df['_'.join(['item-week_shifted-', str(lag)])] = merged_df.groupby(['weekofyear',"物料编码"])["需求量"].transform(lambda x:x.shift(lag).sum())
    merged_df['_'.join(['item-week_shifted-', str(lag)])] = merged_df.groupby(['weekofyear',"物料编码"])["需求量"].transform(lambda x:x.shift(lag).mean()) 
    merged_df['_'.join(['item-week_shifted-', str(lag)])].fillna(merged_df['_'.join(['item-week_shifted-', str(lag)])].mode()[0], inplace=True)
    ##### 需求量 for that item i days in the past
    merged_df['_'.join(['store-week_shifted-', str(lag)])] = merged_df.groupby(['weekofyear',"工厂编码"])["需求量"].transform(lambda x:x.shift(lag).sum())
    merged_df['_'.join(['store-week_shifted-', str(lag)])] = merged_df.groupby(['weekofyear',"工厂编码"])["需求量"].transform(lambda x:x.shift(lag).mean()) 
    merged_df['_'.join(['store-week_shifted-', str(lag)])].fillna(merged_df['_'.join(['store-week_shifted-', str(lag)])].mode()[0], inplace=True)
#%%
demand_data.drop('需求量', axis=1, inplace=True)
merged_df.drop(['过账日期','需求量'], axis=1, inplace=True)
# %%
# comes from the public kernel
# why mult 1?
merged_df = merged_df * 1
params = {
    'nthread': 4,
    'categorical_feature' : [0,1,9,10,12,13,14], # Day, DayOfWeek, Month, Week, Item, Store, WeekOfYear
    'max_depth': 8,
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': 'mape', # this is abs(a-e)/max(1,a)
    'num_leaves': 127,
    'learning_rate': 0.25,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 30,
    'lambda_l1': 0.06,
    'lambda_l2': 0.1,
    'verbose': -1,
}
#%%
'''使用lgb进行stacking'''
import lightgbm as lgb
from sklearn.model_selection import KFold
num_folds = 3
test_x = merged_df[len_train:].values
all_x = merged_df[:len_train].values
all_y = target # removing what we did earlier

oof_preds = np.zeros([all_y.shape[0]])
sub_preds = np.zeros([test_x.shape[0]])

feature_importance_df = pd.DataFrame()
folds = KFold(n_splits=num_folds, shuffle=True, random_state=345665)

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(all_x)):
    
    train_x, train_y = all_x[train_idx], all_y[train_idx]
    valid_x, valid_y = all_x[valid_idx], all_y[valid_idx]
    lgb_train = lgb.Dataset(train_x,train_y)
    lgb_valid = lgb.Dataset(valid_x,valid_y)
        
    # train
    gbm = lgb.train(params, lgb_train, 1000, 
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=100, verbose_eval=100)
    
    oof_preds[valid_idx] = gbm.predict(valid_x, num_iteration=gbm.best_iteration)
    sub_preds[:] += gbm.predict(test_x, num_iteration=gbm.best_iteration) / folds.n_splits
    valid_idx += 1
    importance_df = pd.DataFrame()
    importance_df['feature'] = merged_df.columns
    importance_df['importance'] = gbm.feature_importance()
    importance_df['fold'] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, importance_df], axis=0)
    
e = np.array([0 if np.isnan(i) else i for i in (2 * abs(all_y - oof_preds) / ( abs(all_y)+abs(oof_preds)))])
e = e.mean()
print('Full validation score With Box Cox %.4f' %e)
print('Inverting Box Cox Transformation')
print('Done!!')
#%%
'''进行boxcox转换'''
def inverse_boxcox(y, lambda_):
    return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)
sub_preds = inverse_boxcox(sub_preds , lambda_prophet) - 1
oof_preds = inverse_boxcox(oof_preds , lambda_prophet) - 1
e = 2 * abs(all_y - oof_preds) / ( abs(all_y)+abs(oof_preds) )
print("all_y:%.4f oof_preds%.4f"%(all_y,oof_preds))

e = e.mean()
print('Full validation score Re-Box Cox Transformation is %.4f' %e)
#Don't Forget to apply inverse box-cox
feature_importance_df.sort_values("importance",ascending=False, inplace=True)
# %%
def plot_fi(fi): 
    return fi.plot('feature', 'importance', 'barh', figsize=(12,12), legend=False)
#display(plot_fi(importance_df[:]))
# %%
# one-hot编码
import gc
gc.collect()
print("Before OHE", merged_df.shape)
merged_df = pd.get_dummies(merged_df, columns=['Day', 'Dayofweek', 'Month', 'Week', '物料编码', '工厂编码', 'weekofyear'])
print("After OHE", demand_data.shape)
test_x = merged_df[len_train:].values
all_x = merged_df[:len_train].values
all_y = target
# %%
'''最终回归预测模型'''
def XGB_regressor(train_X, train_y, test_X, test_y= None, feature_names=None, seed_val=2018, num_rounds=500):

    param = {}
    param['objective'] = 'reg:linear'
    param['eta'] = 0.1
    param['max_depth'] = 3
    param['silent'] = 1
    param['eval_metric'] = 'mae'
    param['min_child_weight'] = 4
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['seed'] = seed_val
    # param["tree_method"]="gpu_hist"  是否使用gpu加速
    num_rounds = num_rounds

    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
        
    return model
model = XGB_regressor(train_X = all_x, train_y = all_y, test_X = test_x)
y_test = model.predict(xgb.DMatrix(test_x), ntree_limit = model.best_ntree_limit)
#%%
print('Inverting Box Cox Transformation')
y_test = inverse_boxcox(y_test, lambda_prophet) - 1
#%%
import pickle
with open("./models/y_test.txt","wb+") as f:
    pickle.dump(y_test,f)