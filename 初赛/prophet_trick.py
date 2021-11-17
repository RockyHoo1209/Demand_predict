'''
Author: RockyHoo
Date: 2021-11-17 21:38:32
LastEditTime: 2021-11-17 21:46:13
Description: prophet预测
FilePath: \Demand_predict\prophet_trick.py
'''
import warnings
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
import tqdm
warnings.filterwarnings("ignore")
from numpy.random import seed
from sklearn.multioutput import MultiOutputRegressor
from xgboost.sklearn import XGBClassifier,XGBRegressor
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from chinese_calendar import is_workday, is_holiday
demand_data=pd.read_csv("./data/demand_train.csv")
demand_data['过账日期'] = pd.to_datetime(demand_data['过账日期'])
date_index=pd.date_range("2018-01-01","2021-11-30")
holidays_list=[]
'''选出holiday的日期'''
for date in date_index:
    if is_holiday(date):
        holidays_list.append(date)
        
holidays = pd.DataFrame({
  'holiday' : 'playoff',
  'ds' : pd.to_datetime(holidays_list),
  'lower_window' : 0,
  'upper_window' : 2}
)

date_df=pd.DataFrame()
date_df["过账日期"]=date_index
temp_gp=demand_data.sort_values("过账日期").groupby(["物料编码","工厂编码"],as_index=False)
month_1=[]
month_2=[]
month_3=[]
ans_df=pd.DataFrame()
# demand_filldate=pd.DataFrame()
'''直接遍历每个物料种类进行prophet预测'''
for group in tqdm.tqdm(temp_gp.groups):
    temp_df=temp_gp.get_group(group)
    merged_df=pd.DataFrame()
    merged_df["过账日期"]=date_df["过账日期"]
    merged_df["工厂编码"]=[list(temp_df["工厂编码"])[0]]*len(merged_df)
    merged_df["物料编码"]=[list(temp_df["物料编码"])[0]]*len(merged_df)
    merged_df["物料品牌"]=[list(temp_df["物料品牌"])[0]]*len(merged_df)
    merged_df["物料类型"]=[list(temp_df["物料类型"])[0]]*len(merged_df)
    merged_df["物料品类"]=[list(temp_df["物料品类"])[0]]*len(merged_df)
    merged_df=pd.merge(merged_df,temp_df,how="left").fillna(0)
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True,holidays=holidays)
    prophet_df=merged_df[['过账日期','需求量']]
    prophet_df.columns=["ds","y"]
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=90, include_history=False)
    forecast = m.predict(future)
    temp_ans_df=pd.DataFrame()
    temp_ans_df["工厂编码"]=[list(temp_df["工厂编码"])[0]]
    temp_ans_df["物料编码"]=[list(temp_df["物料编码"])[0]]
    temp_ans_df["M+1月预测需求量"]=[max(0,forecast["yhat"][0:30].sum())]
    temp_ans_df["M+2月预测需求量"]=[max(0,forecast["yhat"][30:60].sum())]
    temp_ans_df["M+3月预测需求量"]=[max(0,forecast["yhat"][60:90].sum())]
    ans_df=ans_df.append(temp_ans_df)
ans_df.to_csv("./data/prophet.csv",index=False)
#%%
