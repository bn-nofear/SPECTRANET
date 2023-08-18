import pickle
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import r2_score,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
import pickle
import datetime
import h5py
import statsmodels.api as sm
import warnings
import time
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.api as smt  
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from arch.unitroot import ADF
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.optimize import minimize
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm, trange
import statsmodels.api as sm
import os

execute_path = '../../result'
data_path = '../../data'

stock_info = pd.read_csv(os.path.join(data_path,'stock_info.csv'),encoding='utf-8-sig')
code_serie = pd.Series(stock_info['证券代码'])
date_serie = pd.Series(pd.read_csv(os.path.join(data_path,'date_serie.csv'),encoding='utf-8-sig',index_col=0).iloc[:,0])
f = h5py.File(os.path.join(data_path,'hs300volume.h5'),'r')
intra_volume = np.array(f['volume'])
f.close()

predict = np.zeros((600,240,280))
for i in range(280):
    predict[:,:,i] = pd.DataFrame(intra_volume[:,:,i]).rolling(20).mean(skipna=True).iloc[-601:-1].values

f = h5py.File(os.path.join(execute_path,'SMA.h5'),'w')
f['output'] = predict.astype(np.float16)
f.close() 