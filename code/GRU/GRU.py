import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
import pickle
import h5py
import warnings
import time
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tsa.api as smt  
import time

#early stopping
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dropout, Dense, LSTM,SimpleRNN,GRU,Conv1D,MaxPooling1D,Flatten,Bidirectional
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
T = time.time()

execute_path = '../../result'
data_path = '../../data'

#load data
f = h5py.File(os.path.join(data_path,'hs300volume.h5'),"r")
intra_volume = np.array(f['volume'])
f.close()

f = h5py.File(os.path.join(data_path,'hs300volume_mask.h5'),"r")
mask_tol = np.array(f['volume_mask'])
f.close()

#set params
num = 280 #num of stocks
train_size = 80
test_size = 20
row_size=20
horizon=1 #prediction length

#Create GRU model
def create_model():
    model = Sequential()
    model.add(GRU(512, input_shape=(1, 240*280),return_sequences=True))
    model.add(GRU(256,return_sequences=True))
    model.add(Dense(240*280))
    model.compile(loss='mse', optimizer='adam')
    return model

model = create_model()


X=intra_volume[-680:,:,:]
mask=mask_tol[-680:,:,:]
y_pred=[]
y_true=[]
#-----------------------------------------------------------------------------Rolling test----------------------------------------------------------------------------------------------------
for j in range(train_size,len(X)-test_size+1,row_size):
    X_train, X_test = X[j-train_size:j,:].copy().reshape((train_size,-1)), X[j-horizon:j+test_size,:].copy().reshape((test_size+horizon,-1))

    scaler1 = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler1.fit_transform(X_train)
    X_test = scaler1.transform(X_test)


    x_train, y_train = X_train[:-horizon, :], X_train[horizon:, :]
    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    x_test, y_test = X_test[:-horizon, :], X_test[horizon:, :]
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
    
    history = model.fit(x_train, y_train, epochs=100, batch_size=8,  verbose=0, shuffle=False,callbacks=[EarlyStopping(monitor='loss', patience=25)])
    test_pred = model.predict(x_test)

    test_pred = test_pred-scaler1.min_
    test_pred = test_pred/scaler1.scale_
    test_true= y_test
    test_true= test_true-scaler1.min_
    test_true = test_true/scaler1.scale_
    test_true=test_true.reshape((test_size,240,280))
    test_pred=test_pred.reshape((test_size,240,280))
    y_true.append(test_true)
    y_pred.append(test_pred)
    print('finished',j)
    
    r2=[]
    for k in range(280):
        r2.append(r2(test_true[:,:,k].reshape(-1),test_pred[:,:,k].reshape(-1)))
    r2=pd.Series(r2)
    print(r2.mean())

y_true=np.array(y_true)
y_pred=np.array(y_pred)

#y_true_save=y_true.reshape((600,240,280))
y_pred_save=y_pred.reshape((600,240,280)).astype('float16')

#save as h5
hf = h5py.File('GRU.h5', 'w')
#hf.create_dataset('Y', data=y_true_save)
hf.create_dataset('output', data=y_pred_save)
hf.close()