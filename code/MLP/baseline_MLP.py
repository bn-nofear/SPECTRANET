import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import sys
import time
from MLP import MLP
from sklearn.metrics import r2_score
import h5py
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm, trange

# NUM,DAYS = sys.argv[1:]

execute_path = '../../result/MLP'
data_path = '../../data'

stock_info = pd.read_csv(os.path.join(data_path,'stock_info.csv'),encoding='utf-8-sig')
code_serie = pd.Series(stock_info['证券代码'])
date_serie = pd.Series(pd.read_csv(os.path.join(data_path,'date_serie.csv'),encoding='utf-8-sig',index_col=0).iloc[:,0])
f = h5py.File(os.path.join(data_path,'hs300volume.h5'),'r')
intra_volume = np.array(f['volume'])
f.close()

# return (batch_size, num, total_length) input_length is the length of the training set, predict_length is the length of the forecasting set. total_length = input_length + predict_length + predict_length
def data_preprossing(num, days, end_day): 
    # --------------------------------------- Generate Data and Predict Data ---------------------------------------
    
    dataslice = intra_volume[end_day-days-5:end_day,:,:num]
    dateslice = date_serie[end_day-days:end_day]
    codeslice = code_serie[:num]

    train_data_x = [dataslice[i:i+5,:,:].swapaxes(0,1).reshape(240,5*num) for i in range(int(days*4/5))]
    train_data_y = [dataslice[i+5,:,:].reshape(240,num) for i in range(int(days*4/5))]
    test_data_x = [dataslice[i:i+5,:,:].swapaxes(0,1).reshape(240,5*num) for i in range(int(days*4/5),days)]
    test_data_y = [dataslice[i+5,:,:].reshape(240,num) for i in range(int(days*4/5),days)]
    train_data_x = torch.from_numpy(np.stack(train_data_x))
    train_data_y = torch.from_numpy(np.stack(train_data_y))
    test_data_x = torch.from_numpy(np.stack(test_data_x))
    test_data_y = torch.from_numpy(np.stack(test_data_y))

   # (batch_size, input_length, num)和(batch_size, output_length, num)
    print('Windowed Train Input shape: ', train_data_x.shape)
    print('Windowed Train Output shape: ', train_data_y.shape)
    print('Windowed Test Input shape: ', test_data_x.shape)
    print('Windowed Test Onput shape: ', test_data_y.shape)

    return train_data_x, train_data_y, test_data_x, test_data_y, dateslice, codeslice

def our_experiment(num, days, end_day):
    os.chdir(execute_path)
    os.makedirs('{}--{}--{}'.format(num,days,end_day),exist_ok=True)
    os.chdir('{}--{}--{}'.format(num,days,end_day))

    Time = []

    # --------------------------------------- Set parameters ---------------------------------------
    learning_rate = 1e-3
    iterations = 100
    in_feature = int(5*num)
    out_feature = int(num)
    # n_layers = 2

    # --------------------------------------- Data processing ---------------------------------------
    print("Data processing...")
    train_data_x, train_data_y, test_data_x, test_data_y, dateslice, codeslice = data_preprossing(num, days, end_day)
    dateslice.to_csv("date.csv")
    codeslice.to_csv("code.csv")
    # batch_size = train_data_x.shape[0]

    # --------------------------------------- Training ---------------------------------------
    model = MLP(in_feature,out_feature)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_func = nn.MSELoss()
    mse = []
    print("Training...")
    # available_mask = torch.ones(train_data.shape) 
    # 训练
    for epoch in tqdm(range(iterations)):
        output = model(train_data_x.float()) 
        loss = loss_func(output.float(),train_data_y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mse.append(loss.item())


    # --------------------------------------- Training Analysis ---------------------------------------
    # os.makedirs('Train',exist_ok=True)

    # output = output.detach().numpy().astype(np.float16)
    # Y = train_data_y.detach().numpy().astype(np.float16)

    # f = h5py.File(r'Train\\result.h5', 'w') 
    # f['output'] = output
    # f['Y'] = Y
    # f.close()  

    # --------------------------------------- Testing ---------------------------------------
    output = model(test_data_x.float())

    # --------------------------------------- Testing Analysis ---------------------------------------
    # os.makedirs('Test',exist_ok=True)

    output = output.detach().numpy().astype(np.float16)
    Y = test_data_y.detach().numpy().astype(np.float16)

    f = h5py.File(r'result.h5', 'w') 
    f['output'] = output
    f['Y'] = Y
    f.close() 

    
if __name__ == "__main__":
    for end_day in range(len(date_serie)-580,len(date_serie)+20,20):
        our_experiment(280,100,end_day=end_day)