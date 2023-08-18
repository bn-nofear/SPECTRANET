import h5py
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import datetime as dt
import numpy as np
from sklearn.metrics import r2_score

execute_path = '../../result'
data_path = '../../data'

# Data loading
f = h5py.File(os.path.join(data_path,'hs300volume.h5'),"r")
intra_volume = np.array(f['volume'])
f.close()
intra_volume=intra_volume[-680:,:,:]
print(intra_volume.shape)

#shift to float32
def data_adjust(data):
    data = data.swapaxes(0,2).swapaxes(1,2)
    data = data.astype(np.float32)
    return data

intra_volume = data_adjust(intra_volume)


def VAR_model(data,train_size=80,test_size=20):
    
    num=data.shape[0]
    models=[]
    for i in range(240):
    
    #factor computation
        train_data=data[:,:train_size,i]
        model=sm.tsa.VAR(train_data.T)
        model_fit=model.fit(5)
        models.append(model_fit)

    print('model fitting is done!')
    print('start prediction!')
    #prediction
    pred=np.zeros((num,test_size,240))
    for j in range(test_size):
        known=data[:,:train_size+j,:]
        for i in range(240):
            known_data=known[:,:,i].T
            bin_pred=models[i].forecast(y=known_data,steps=1)
            pred[:,j,i]=bin_pred
           
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
    volume_pred[:,i:i+20]=VAR_model(intra_volume[:,i:(i+100),:],train_size=80,test_size=20)
    print(str(i)+'th day is done!')
    print('end time is '+str(dt.datetime.now()))
    
volume_pred=volume_pred.swapaxes(1,2).swapaxes(0,2)
print(volume_pred.shape)
f = h5py.File(os.path.join(execute_path,'VAR.h5'),'w')
f.create_dataset('test_pred',data=volume_pred)
f.close()
