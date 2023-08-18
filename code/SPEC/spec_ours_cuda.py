import os
import pickle
import numpy as np
import pandas as pd
import sys
from sklearn.metrics import r2_score,mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
import pickle
import datetime
import h5py
import torch
import torch.nn as nn
from spectranetimport _TemporalModel
import time
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
STOCK_LIST = [i for i in range(280)]
DAYS,n_time_in,n_time_out = sys.argv[1:]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

execute_path = '../../result'
data_path = '../../data'

# read data
f = h5py.File(os.path.join(data_path,'hs300volume.h5'),"r")
intra_volume = np.array(f['volume'])
f.close()
f = h5py.File(os.path.join(data_path,'hs300volume_mask.h5'),"r")
missing_mask= np.array(f['volume_mask'])
f.close()

stock_info = pd.read_csv(os.path.join(data_path,'stock_info.csv'),encoding='utf-8-sig')
code_serie = pd.Series(stock_info['证券代码'])
date_serie = pd.Series(pd.read_csv(os.path.join(data_path,'date_serie.csv'),encoding='utf-8-sig',index_col=0).iloc[:,0])

# return (batch_size, num, total_length) input_length is the length of the training set, predict_length is the length of the forecasting set. total_length = input_length + predict_length + predict_length
def data_preprossing(stock_list, days, n_time_in, n_time_out , end_day): 
    # --------------------------------------- Generate Data and Predict Data ---------------------------------------
    
    dataslice = intra_volume[end_day-days-int(n_time_in/240):end_day,:,stock_list]
    maskslice = missing_mask[end_day-days-int(n_time_in/240):end_day,:,stock_list]
    dateslice = date_serie.iloc[end_day-days:end_day]
    codeslice = code_serie.iloc[stock_list]
    
    res = np.zeros([days,len(stock_list),n_time_in+n_time_out])
    res_mask = np.zeros([days,len(stock_list),n_time_in+n_time_out])

    for i in range(days):
        temp = dataslice[i:i+int((n_time_in+n_time_out)/240),:,:]
        temp_mask = maskslice[i:i+int((n_time_in+n_time_out)/240),:,:]

        res[i] = temp.reshape(-1,temp.shape[-1]).swapaxes(0,1)
        res_mask[i] = temp_mask.reshape(-1,temp_mask.shape[-1]).swapaxes(0,1)
    
    train_data = torch.Tensor(res[:-int(1/5*res.shape[0])])
    test_data = torch.Tensor(res[-int(1/5*res.shape[0]):])
    train_missing_mask = torch.Tensor(res_mask[:-int(1/5*res.shape[0])])
    test_missing_mask = torch.Tensor(res_mask[-int(1/5*res.shape[0]):])
    
    train_data_mask = torch.ones(train_data.shape)
    test_data_mask = torch.ones(test_data.shape)
    train_data_mask[:,:,n_time_in:] = 0
    test_data_mask[:,:,n_time_in:] = 0

    print('Windowed Train shape: ', train_data.shape)
    print('Windowed Test shape: ', test_data.shape)
    print("Train mask shape:", train_data_mask.shape)
    print("Test mask shape:", test_data_mask.shape)
    print("Train missing mask shape:", train_missing_mask.shape)
    print("Test missing mask shape:", test_missing_mask.shape)

    # Migrating data to the GPU
    train_data_cuda = train_data.cuda()
    test_data_cuda = test_data.cuda()
    train_data_mask_cuda = train_data_mask.cuda()
    test_data_mask_cuda = test_data_mask.cuda()

    return train_data_cuda, test_data_cuda, train_data_mask_cuda, test_data_mask_cuda, dateslice, codeslice, train_missing_mask, test_missing_mask


def our_experiment(stock_list, days, n_time_in, n_time_out, end_day):
    os.chdir(execute_path)
    os.makedirs('{}--{}--{}'.format(len(stock_list),days,end_day),exist_ok=True)
    os.chdir('{}--{}--{}'.format(len(stock_list),days,end_day))

    Time = []

    # --------------------------------------- Set parameters ---------------------------------------
    learning_rate = 1e-3
    iterations = 80
    n_features = len(stock_list)
    n_polynomial = 5 
    n_harmonics = 1 
    z_iters = 30
    z_sigma = 0.25
    z_step_size = 0.1
    step = 240
    normalize_windows = False
    z_t_dim = int((n_time_in + n_time_out))
    stride = 4
    kernel_size = 8
    max_filters = 512
    n_filters_multiplier = 64
    dilation = 1
    z_with_noise = False
    z_persistent = False
    n_layers = 3

    # --------------------------------------- Data processing ---------------------------------------
    print("Data processing...")
    T = time.time()
    train_data, test_data, train_data_mask, test_data_mask, dateslice, codeslice,train_missing_mask, test_missing_mask= data_preprossing(stock_list, days, n_time_in, n_time_out, end_day)
    train_missing_mask=train_missing_mask.to(device=device)
    test_missing_mask=test_missing_mask.to(device=device)
    dateslice.to_csv("date.csv")
    codeslice.to_csv("code.csv")
    Time.append(time.time() - T)
    print("Time taken:.{}".format(time.time() - T))

    # --------------------------------------- Training ---------------------------------------
    model = _TemporalModel(n_time_in=n_time_in, n_time_out=n_time_out, univariate=False,
                                    n_features=n_features, n_layers=n_layers, n_filters_multiplier=n_filters_multiplier, max_filters=max_filters,
                                    kernel_size=kernel_size, stride=stride, dilation=dilation,
                                    normalize_windows=normalize_windows,
                                    z_t_dim=z_t_dim, n_polynomial=n_polynomial, n_harmonics=n_harmonics, z_iters=z_iters, z_sigma=z_sigma, z_step_size=z_step_size,
                                    z_with_noise=z_with_noise, z_persistent=z_persistent)
    # Migrating data to the gpu
    
    model.to(device=device)
    #print(torch.cuda.memory_allocated())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # loss_func = nn.MSELoss()
    loss_func = nn.MSELoss()
    mse = []
    print("Training...")
    T = time.time()
    # available_mask = torch.ones(train_data.shape)  # Whether to randomly mask the training data
    # Traning
    for epoch in range(iterations):
        #print(epoch)
        Y, output, z = model(train_data, train_data_mask)
        #output = output * train_missing_mask
        loss = loss_func(output,Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mse.append(loss.item())
    Time.append(time.time() - T)
    #print("Time taken:.{}".format(time.time() - T))
   
    
    # --------------------------------------- Training Analysis ---------------------------------------
    os.makedirs('Train',exist_ok=True)
    # Save the model
    # torch.save(model.state_dict(), 'Train/model.pt')

    # Migrate y, output, z, etc. back to the CPU
    Y_cpu, output_cpu, z_cpu = Y.cpu(), output.cpu(), z.cpu()

    # Reduce to a sequence of [stock, time]
    output_cpu = output_cpu[:,:,-n_time_out:].swapaxes(1,2).detach().numpy().astype(np.float16)
    Y_cpu = Y_cpu[:,:,-n_time_out:].swapaxes(1,2).detach().numpy().astype(np.float16)

    f = h5py.File('Train/result.h5', 'w') 
    f['output'] = output_cpu
    f['Y'] = Y_cpu
    f.close()  
    # pd.DataFrame(z_cpu.detach().numpy().astype(np.float16)).to_hdf('Train/train-latent variables.h5',key='benyan')

    # --------------------------------------- Testing ---------------------------------------
    print("Testing...")
    T = time.time()
    Y, output, z= model(test_data, test_data_mask)
    #output = output * test_missing_mask
    Time.append(time.time() - T)
    print("Time taken:.{}".format(time.time() - T))

    # --------------------------------------- Testing Analysis ---------------------------------------
    os.makedirs('Test',exist_ok=True)

    # Migrate y, output, z, etc. back to the CPU
    Y_cpu, output_cpu, z_cpu = Y.cpu(), output.cpu(), z.cpu()

    output_cpu = output_cpu[:,:,-n_time_out:].swapaxes(1,2).detach().numpy().astype(np.float16)
    Y_cpu = Y_cpu[:,:,-n_time_out:].swapaxes(1,2).detach().numpy().astype(np.float16)

    f = h5py.File('Test/result.h5', 'w') 
    f['output'] = output_cpu
    f['Y'] = Y_cpu
    f.close() 
    # pd.DataFrame(z_cpu.detach().numpy().astype(np.float16)).to_hdf('Test/test-latent variables.h5',key='benyan')
    
    r2=[]
    for i in range(output_cpu.shape[2]):
        r2.append(r2_score(Y_cpu[:,:,i].reshape(-1),output_cpu[:,:,i].reshape(-1)))
    r2=np.array(r2)
    print('r2:',r2.mean())
    # Describe the r2 distribution, including median, quantile
    r2=pd.Series(r2)
    print(r2.describe())

    plt.figure()
    plt.plot([i for i in range(iterations)], mse)
    pd.Series(mse).to_csv('mse.csv')
    plt.savefig('mse.jpg')
    pd.DataFrame(Time,index=['Data-Processing','Training','Testing']).to_csv('Time.csv')


for end_day in range(len(date_serie)-580,len(date_serie)+20,20):
    our_experiment(stock_list=STOCK_LIST,days=int(DAYS),n_time_in=int(n_time_in),n_time_out=int(n_time_out),end_day=end_day)