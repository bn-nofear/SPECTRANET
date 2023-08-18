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
import time
import matplotlib.pyplot as plt
import argparse
from models.Informer import Model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

def set_configs():
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')
    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    return args

def our_experiment(stock_list, days, n_time_in, n_time_out, end_day):
    os.chdir(execute_path)
    os.makedirs('{}--{}--{}'.format(len(stock_list),days,end_day),exist_ok=True)
    os.chdir('{}--{}--{}'.format(len(stock_list),days,end_day))

    Time = []
    iterations = 80
    learning_rate = 0.0001

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
    configs = set_configs()
    model = Model(configs)
    # Migrating data to the gpu
    
    model.to(device=device)
    #print(torch.cuda.memory_allocated())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # loss_func is changed to 1-r2 along the third axis, then averaged
    class loss_func_daily(nn.Module):
        def __init__(self):
            super(loss_func_daily,self).__init__()
        
        def forward(self,x,y):
            r2_list=[]
            def r2_loss(x,y):
                y_var=torch.mean(torch.pow((y - torch.mean(y)), 2))
                loss_score=torch.mean(torch.pow((x - y), 2))/y_var
                return loss_score
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    r2_list.append(r2_loss(x[i,j,:],y[i,j,:]))
            r2_list=torch.Tensor(r2_list)
            return r2_list.mean()

    loss_func =nn.MSELoss()
    mse = []
    print("Training...")
    T = time.time()
    # available_mask = torch.ones(train_data.shape) # Whether to randomly mask the training data
    # data processing (for transformer only)
    days = int((n_time_in+n_time_out)/240)
    batch_size = train_data.shape[0]
    batch_x = train_data[:,:,:n_time_in]
    batch_y = train_data[:,:,-n_time_out:]
    dec_inp = torch.zeros_like(batch_y).float()
    dec_inp = torch.cat([batch_x[:,:,-(n_time_in-n_time_out):], dec_inp],dim=-1).float().to(device).permute(0,2,1)
    enc_inp = batch_x.permute(0,2,1)
    day_stamp = torch.Tensor([i for i in range(days)]).repeat_interleave(240)[None,:].repeat(batch_size, dim=0)[:,:,None]
    tick_stamp = torch.Tensor([i for i in range(240)]).repeat(days)[None,:].repeat(batch_size, dim=0)[:,:,None]
    time_stamp = torch.cat([day_stamp, tick_stamp], dim=-1).float().to(device)
    # Training
    for epoch in range(iterations):
        mark_enc = time_stamp[:,:n_time_in,:]
        mark_dec = time_stamp[:,n_time_out:,:]

        output = model(enc_inp, mark_enc, dec_inp, mark_dec)
        print(output.shape)
        print(batch_y.shape)
        Y_weight= batch_y
        output = output
        output = output * train_missing_mask.permute(0,2,1)[:,:,n_time_in:]
        loss = loss_func(output,Y_weight)
        optimizer.zero_grad()
        print(torch.autograd.set_detect_anomaly(True))
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
    Y_cpu, output_cpu = Y_weight.cpu(), output.cpu()

    # Reduce to a sequence of [stock, time]
    output_cpu = output_cpu[:,-n_time_out:,:].permute(2,0,1).reshape(280,-1).detach().numpy().astype(np.float16)
    Y_cpu = Y_cpu[:,-n_time_out:,:].permute(2,0,1).reshape(280,-1).detach().numpy().astype(np.float16)

    f = h5py.File('Train/result.h5', 'w') 
    f['output'] = output_cpu
    f['Y'] = Y_cpu
    f.close()  
    # pd.DataFrame(z_cpu.detach().numpy().astype(np.float16)).to_hdf('Train/train-latent variables.h5',key='benyan')

    # --------------------------------------- Testing ---------------------------------------
    # data processing (for transformer only)
    batch_size = test_data.shape[0]
    batch_x = test_data[:,:,:n_time_in]
    batch_y = test_data[:,:,-n_time_out:]
    dec_inp = torch.zeros_like(batch_y).float()
    dec_inp = torch.cat([batch_x[:,:,-(n_time_in-n_time_out):], dec_inp],dim=-1).float().to(device).permute(0,2,1)
    enc_inp = batch_x.permute(0,2,1)
    day_stamp = torch.Tensor([i for i in range(days)]).repeat_interleave(240)[None,:].repeat(batch_size, dim=0)[:,:,None]
    tick_stamp = torch.Tensor([i for i in range(240)]).repeat(days)[None,:].repeat(batch_size, dim=0)[:,:,None]
    time_stamp = torch.cat([day_stamp, tick_stamp], dim=-1).float().to(device)
    print("Testing...")
    T = time.time()
    output = model(enc_inp, mark_enc, dec_inp, mark_dec)
    output = output * test_missing_mask.permute(0,2,1)[:,:,n_time_in:]
    Time.append(time.time() - T)
    print("Time taken:.{}".format(time.time() - T))

    # --------------------------------------- Testing Analysis ---------------------------------------
    os.makedirs('Test',exist_ok=True)

    # Migrate y, output, z, etc. back to the CPU
    Y_cpu, output_cpu = Y_weight.cpu(), output.cpu()

    output_cpu = output_cpu[:,-n_time_out:,:].permute(2,0,1).reshape(280,-1).detach().numpy().astype(np.float16)
    Y_cpu = Y_cpu[:,-n_time_out:,:].permute(2,0,1).reshape(280,-1).detach().numpy().astype(np.float16)

    f = h5py.File('Test/result.h5', 'w') 
    f['output'] = output_cpu
    f['Y'] = Y_cpu
    f.close() 
    # pd.DataFrame(z_cpu.detach().numpy().astype(np.float16)).to_hdf('Test/test-latent variables.h5',key='benyan')
    
    r2=[]
    for i in range(output_cpu.shape[0]):
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