import h5py
import os
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.metrics import r2_score
import statsmodels.api as sm

execute_path = '../../result/'
data_path = '../../data'

# read data
f = h5py.File(os.path.join(data_path,'hs300volume.h5'),"r")
intra_volume = np.array(f['volume'])
f.close()
intra_volume=intra_volume[-680:,:,:]
print(intra_volume.shape)

def data_adjust(data):
    data = data.swapaxes(0,2).swapaxes(1,2)
    data = data.reshape(data.shape[0],-1)
    data = data.astype(np.float32)
    return data

intra_volume = data_adjust(intra_volume)

#compute r-largest eigenvector , type=float32
def get_eigenvector(data,r):
    U,sigma,VT = np.linalg.svd(data,full_matrices=False)
    return VT[:r,:]


def factor_model(data,r,L=20,train_size=80,test_size=20):
    
    num=data.shape[0]
    
    #factor computation
    train_data=data[:,:train_size*240]#train_data.shape=(num,train_size*240)
    test_data=data[:,train_size*240:]#test_data.shape=(num,test_size*240)
    
    FT=get_eigenvector(train_data,r)#FT.shape=(r,T)

    Lambda=np.dot(FT,train_data.T) #Lambda.shape=(r,num)

    factor=np.dot(FT.T,Lambda).T#factor.shape=(num,T)

    residue=train_data-factor #residue.shape=(num,T)
    factor_interday=factor.reshape((num,train_size,240))
    
    #residue model
    res_model=[]
    for i in range(num):
        model=sm.tsa.ARIMA(residue[i],order=(1,0,1))
        model_fit=model.fit()
        res_model.append(model_fit)
        #print(str(i)+'th model is fitted!')

    print('model fitting is done!')
    print('start prediction!')
    #prediction
    pred=np.zeros((num,test_size,240))
    for j in range(test_size):
        factor_day_pred=np.zeros((num,1,240))
        res_day_updated=np.zeros((num,240))

        for i in range(num):
            #pred residue 
            res_model_updated=sm.tsa.ARIMA(residue[i],order=(1,0,1))
            res_model_updated_fit=res_model_updated.filter(res_model[i].params)
            res_pred=res_model_updated_fit.predict(start=train_size*240+j*240,end=train_size*240+(j+1)*240-1)

            # pred factor through the same bin mean in last L days 
            factor_pred=factor_interday[i,-L:,:].mean(axis=0)
            factor_day_pred[i,0]=factor_pred

            # pred volume
            pred[i,j]=factor_pred+res_pred
           
            new_residue=test_data[i,j*240:(j+1)*240]-factor_pred
            res_day_updated[i]=new_residue

        #update residue 
        residue=np.concatenate((residue,res_day_updated),axis=1)    

        #update factor_interday
        factor_interday=np.concatenate((factor_interday,factor_day_pred),axis=1)
    
    print('prediction is done!')
    
    return pred


days=680
vol_pred=np.zeros((283,days-80,240))

    
volume_data=intra_volume
stock_num=volume_data.shape[0]
# rolling 100 days 
volume_pred=np.zeros((stock_num,days-80,240))
for i in range(0,days-100+1,20):
    print('start '+str(i)+'th day')
    print('start time is '+str(dt.datetime.now()))
    volume_pred[:,i:i+20]=factor_model(intra_volume[:,i*240:(i+100)*240],r=3,L=20,train_size=80,test_size=20)
    print(str(i)+'th day is done!')
    print('end time is '+str(dt.datetime.now()))
    if i==0:
        print('r2 is '+str(r2_score(intra_volume[0,80*240:100*240].reshape(-1),volume_pred[0,:20,:].reshape(-1))))
f = h5py.File(os.path.join(execute_path,'SVD.h5'),'w')
f.create_dataset('test_pred',data=volume_pred)
f.close()