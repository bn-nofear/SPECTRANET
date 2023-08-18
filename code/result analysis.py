import os
import pickle
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import r2_score,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
import pickle
import datetime
import h5py
import warnings
warnings.filterwarnings("ignore")
import copy
import matplotlib.pyplot as plt
import seaborn as sns

execute_path = '../../result'
data_path = '../../data'
result_path = '../../result'

os.makedirs(result_path,exist_ok=True)

stock_info = pd.read_csv(os.path.join(data_path,'stock_info.csv'),encoding='utf-8-sig')
code_serie = pd.Series(stock_info['证券代码'])
date_serie = pd.Series(pd.read_csv(os.path.join(data_path,'date_serie.csv'),encoding='utf-8-sig',index_col=0).iloc[:,0])

f = h5py.File(os.path.join(result_path,"result.h5"),"r")
predict = np.array(f['output'])
f.close()

stock_info = pd.read_csv(os.path.join(data_path,'stock_info.csv'),encoding='utf-8-sig')
code_serie = pd.Series(stock_info['证券代码'])
date_serie = pd.Series(pd.read_csv(os.path.join(data_path,'date_serie.csv'),encoding='utf-8-sig',index_col=0).iloc[:,0])
f = h5py.File(os.path.join(data_path,'hs300close.h5'),"r")
intra_close = np.array(f['close'])
f.close()
f = h5py.File(os.path.join(data_path,'hs300volume.h5'),'r')
intra_volume = np.array(f['volume'])
f.close()

test_close = intra_close[-600:,:,:]
test_volume_true = intra_volume[-600:,:,:]
test_volume_predict = predict

# Determine whether the day is available for trading
validday = (test_volume_true.sum(axis=1)!=0)
valid = np.stack([validday for i in range(240)],axis=2).swapaxes(1,2)
validnum = pd.DataFrame(validday.astype(int))

test_volume_predict = valid * test_volume_predict

# total analysis
num = 280
total_true = test_volume_true.swapaxes(0,2).swapaxes(1,2).reshape(num,-1)
total_predict =  test_volume_predict.swapaxes(0,2).swapaxes(1,2).reshape(num,-1)
total_valid = valid.swapaxes(0,2).swapaxes(1,2).reshape(num,-1)

vwap_true = pd.DataFrame(index=date_serie[-test_volume_true.shape[0]:],columns=code_serie)
vwap_predict = pd.DataFrame(index=date_serie[-test_volume_true.shape[0]:],columns=code_serie)
for i in range(test_volume_true.shape[-1]):
    vwap_true[code_serie.iloc[i]] = [((test_volume_true[j,:,i] * test_close[j,:,i]).sum() / test_volume_true[j,:,i].sum()) for j in range(test_volume_true.shape[0])]
    vwap_predict[code_serie.iloc[i]] = [((test_volume_predict[j,:,i] * test_close[j,:,i]).sum() / test_volume_predict[j,:,i].sum()) for j in range(test_volume_predict.shape[0])]

total_result = pd.DataFrame(index=['R2','MSE','MAE','CORR','VWAP-MAPE-%'])
for i in range(num):
    total_result[code_serie.iloc[i]] = [r2_score(total_true[i,total_valid[i]],total_predict[i,total_valid[i]]),
                                        mean_squared_error(total_true[i,total_valid[i]],total_predict[i,total_valid[i]]),
                                        mean_absolute_error(total_true[i,total_valid[i]],total_predict[i,total_valid[i]]),
                                        np.corrcoef(total_true[i,total_valid[i]],total_predict[i,total_valid[i]])[0,1],
                                        100*mean_absolute_percentage_error(vwap_true.iloc[:,i].loc[vwap_predict.iloc[:,i].dropna().index],vwap_predict.iloc[:,i].dropna())
                                        ]    
total_result.to_csv(os.path.join(result_path,"total_result.csv"))

# intraday analysis
intraday_result = pd.DataFrame(index=date_serie[-test_volume_true.shape[0]:],columns=code_serie)
for i in range(test_volume_true.shape[-1]):
    intraday_result[code_serie.iloc[i]] = [r2_score(test_volume_true[j,:,i],test_volume_predict[j,:,i]) for j in range(test_volume_true.shape[0])]

intraday_result.replace(1,np.nan,inplace=True)
intraday_result.to_hdf(os.path.join(result_path,"intraday_result.h5"),key="benyan")

test_volume_predict_new = np.zeros(shape=test_volume_predict.shape)
a = test_volume_predict.sum(axis=1)
for i in range(test_volume_predict_new.shape[0]):
    for j in range(test_volume_predict_new.shape[2]):
        test_volume_predict_new[i,:,j] = test_volume_predict[i,:,j] / a[i,j]

test_volume_true_new = np.zeros(shape=test_volume_true.shape)
a = test_volume_true.sum(axis=1)
for i in range(test_volume_true_new.shape[0]):
    for j in range(test_volume_true_new.shape[2]):
        test_volume_true_new[i,:,j] = test_volume_true[i,:,j] / a[i,j]

def mae(a,b):
    try:
        return(mean_absolute_error(a,b))
    except:
        return(np.nan)
    
AD_result = pd.DataFrame(index=date_serie[-test_volume_true.shape[0]:],columns=code_serie)
for i in range(test_volume_true_new.shape[-1]):
    AD_result[code_serie.iloc[i]] = [100*mae(test_volume_true_new[j,:,i],test_volume_predict_new[j,:,i]) for j in range(test_volume_true_new.shape[0])]
AD_result.to_hdf(os.path.join(result_path,"AD_result.h5"),key="benyan")
