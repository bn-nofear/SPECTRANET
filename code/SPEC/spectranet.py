import time
import numpy as np
import pandas as pd
import random
import h5py
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
# import pytorch_lightning as pl
from torch import optim

data_path =  '../../data'
f = h5py.File("../../data/hs300historyvolume.h5","r")
history_volume = np.array(f['volume'])
f.close()

def instantiate_trend_basis(degree_of_polynomial, size):
    repeat = int(size/240)
    polynomial_size = degree_of_polynomial + 1
    forward_basis = torch.tensor(np.concatenate([np.power(np.arange(240, dtype=float) / 240, i)[None, :]
                                            for i in range(1,polynomial_size)]), dtype=torch.float32)
    reverse_basis = torch.tensor(np.concatenate([np.power(1 - np.arange(240, dtype=float) / 240, i)[None, :]
                                            for i in range(1,polynomial_size)]), dtype=torch.float32)
    basis = torch.cat([forward_basis, reverse_basis], dim=0)
    final_basis = torch.hstack([basis for i in range(repeat)])

    return final_basis

def instantiate_seasonality_basis(size):
    repeat = int(size/240)
    # frequency = np.array([1,2,3,4,5,6,8,10,12,15,16,20,24,30,40,48,60,80,120,240])
    frequency = np.array([i/3 for i in range(360)])

    forecast_grid = 2 * np.pi * ( np.arange(240, dtype=float)[:, None] / 240) * frequency

    cos_basis = torch.tensor(np.transpose(np.cos(forecast_grid)), dtype=torch.float32)
    sin_basis = torch.tensor(np.transpose(np.sin(forecast_grid)), dtype=torch.float32)
    basis = torch.cat([cos_basis, sin_basis], dim=0)
    final_basis = torch.hstack([basis for i in range(repeat)])

    return final_basis

def instantiate_unique_basis(size):
    repeat = int(size/240)
    pos = [i for i in range(15)] + [i for i in range(120-15,120+15)] + [i for i in range(240-15,240)]
    
    basis = torch.zeros([60,240], dtype=torch.float32)
    for i in range(60):
        basis[i,pos[i]] = 1
       
    final_basis = torch.hstack([basis for i in range(repeat)])

    return final_basis

def instantiate_history_basis1(size):
    repeat = int(size/240)
    basis = torch.from_numpy(history_volume.mean(axis=0).swapaxes(0,1))
    final_basis = torch.hstack([basis for i in range(repeat)])
    return final_basis

def instantiate_history_basis2(size):
    repeat = int(size/240)
    basis = torch.from_numpy(history_volume.mean(axis=2))
    final_basis = torch.hstack([basis for i in range(repeat)])
    return final_basis



class TrimCenterLayer(nn.Module):
    def __init__(self, output_size):
        """
        Select output_size center along last dimension.
        """
        super(TrimCenterLayer, self).__init__()        
        self.output_size = output_size

    def forward(self, x):
        input_size = x.shape[-1]
        assert input_size >= self.output_size, f'Input size {input_size} is not long enough for {self.output_size}.'

        init = (input_size - self.output_size)//2

        return x[..., init:(init+self.output_size)]


class Generator(nn.Module):
    def __init__(self, window_size, n_filters_multiplier, z_t_dim, n_polynomial, n_layers, n_features, max_filters, kernel_size, stride, dilation, output):
        super(Generator, self).__init__()

        # Basis
        trend_basis = instantiate_trend_basis(degree_of_polynomial=n_polynomial, size=z_t_dim)
        seasonality_basis = instantiate_seasonality_basis(size=z_t_dim)
        unique_basis = instantiate_unique_basis(size=z_t_dim)
        #seasonality_basis_2 = seasonality_basis_1(size=z_t_dim)
        #history_basis1 = instantiate_history_basis1(size=z_t_dim)
        #history_basis2 = instantiate_history_basis2(size=z_t_dim)
        self.basis = nn.Parameter(torch.cat([trend_basis,seasonality_basis,unique_basis], dim=0), requires_grad=False)
        print(trend_basis.shape,seasonality_basis.shape,unique_basis.shape,self.basis.shape)
        self.z_d_dim = len(self.basis)


        # n_layers = int(np.log2(window_size/z_t_dim))+1
        n_layers = n_layers
        layers = []
        filters_list = [self.z_d_dim]
        output_size = output
        
        # Hidden layers
        for i in range(0, n_layers):
            filters = min(max_filters, n_filters_multiplier*(2**(n_layers-i-1)))
            layers.append(nn.ConvTranspose1d(in_channels=filters_list[-1], out_channels=filters, dilation=dilation,
                                             kernel_size=kernel_size, stride=stride, padding=1, bias=False))
            layers.append(nn.BatchNorm1d(filters))
            #leaky
            layers.append(nn.ELU())

            # Increase temporal dimension from layer1 
            if i > 0:
                output_size *= 2
            layers.append(TrimCenterLayer(output_size=output_size))
            filters_list.append(filters)

        # Output layer
        layers.append(nn.ConvTranspose1d(in_channels=filters_list[-1], out_channels=n_features, dilation=dilation,
                                         kernel_size=kernel_size, stride=stride, padding=1, bias=False))
        layers.append(TrimCenterLayer(output_size=output))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        # Basis expansion
        x = x[:,:,None]*self.basis[None, :, :]

        # ConvNet
        for layer in self.layers:
            x = layer(x)

        return x


class _TemporalModel(nn.Module):
    def __init__(self, n_time_in, n_time_out,
                 n_features, univariate, n_layers, n_filters_multiplier, max_filters, kernel_size, stride, dilation,
                 z_t_dim, n_polynomial, n_harmonics, z_iters, z_sigma, z_step_size, z_with_noise, z_persistent,
                 normalize_windows):
        super().__init__()

        # Data
        self.n_time_in = n_time_in
        self.n_time_out = n_time_out
        self.window_size = n_time_in+n_time_out
        self.univariate = univariate

        # Generator
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_filters_multiplier = n_filters_multiplier
        self.z_t_dim = z_t_dim
        self.n_polynomial = n_polynomial
        self.n_harmonics = n_harmonics
        self.max_filters = max_filters
        self.kernel_size = kernel_size
        self.stride=stride
        self.dilation=dilation
        self.normalize_windows = normalize_windows

        # Alternating back-propagation
        self.z_iters = z_iters
        self.z_sigma = z_sigma
        self.z_step_size = z_step_size
        self.z_with_noise = z_with_noise
        self.z_persistent = z_persistent

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.univariate:
            n_features = 1

        self.generator = Generator(window_size=self.window_size,
                                   n_features=n_features,
                                   z_t_dim=self.z_t_dim,
                                   n_polynomial=self.n_polynomial,
                                #    n_harmonics=self.n_harmonics,
                                   n_layers=self.n_layers,
                                   n_filters_multiplier=self.n_filters_multiplier,
                                   max_filters=self.max_filters,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride, dilation=self.dilation, output=n_time_in + n_time_out)

    def infer_z(self, z, Y, mask, n_iters, with_noise):

        step_size = self.z_step_size
        n_steps = int(np.ceil(n_iters/3))

        # Protection for 0 n_iters
        if n_iters == 0:
            z = torch.autograd.Variable(z, requires_grad=True)
            
        for i in range(n_iters):
            #print('z_iter',i)
            z = torch.autograd.Variable(z, requires_grad=True)
            #print("before",torch.cuda.memory_allocated())
            Y_hat = self.generator(z)
            mse = (Y_hat-Y)**2

            normalizer = torch.sum(mask/len(Y))
            L = 1.0 / (2.0 * self.z_sigma * self.z_sigma) * (torch.sum(mask*mse)/normalizer) # (Y.shape[1]*Y.shape[2])
            L.backward()
            
            z = z - 0.5 * step_size * (z + z.grad) # self.z_step_size
            if with_noise:
                eps = torch.randn(z.shape).to(z.device)
                z += step_size * eps
            if (i % n_steps == 0) and (i>0):
                step_size = step_size*0.5 

            z = z.detach()

        return z

    def sample_gaussian(self, shape):
        return torch.normal(0, 0.001, shape)

    def _load_current_chain(self, idxs):

        p_0_z = self.p_0_chains[idxs].to(idxs.device)

        if self.univariate:
            p_0_z = p_0_z.reshape(len(p_0_z)*self.n_features,self.generator.z_d_dim)
       
        return p_0_z

    def forward(self, Y, mask, idxs=None, z_0=None):
        
        # if self.univariate:
        #     initial_batch_size, n_features, t_size = Y.shape
        #     Y = Y.reshape(initial_batch_size*n_features, 1, t_size)
        #     mask = mask.reshape(initial_batch_size*n_features, 1, t_size)
        
        batch_size = len(Y)

        if self.normalize_windows:
            # Masked mean and std
            sum_mask = torch.sum(mask, dim=2, keepdims=True)
            mask_safe = sum_mask.clone()
            mask_safe[mask_safe==0] = 1

            sum_window = torch.sum(Y, dim=2, keepdims=True)
            mean = sum_window/mask_safe

            sum_square = torch.sum((mask*(Y-mean))**2, dim=2, keepdims=True)
            std = torch.sqrt(sum_square/mask_safe)

            mean[sum_mask==0] = 0.0
            std[sum_mask==0] = 1.0
            std[std==0] = 1.0

            Y = (Y-mean)/std

        # if (self.z_persistent) and (idxs is not None) and (z_0 is None):
        #     z_0 = self._load_current_chain(idxs=idxs)
        # elif (z_0 is None):
        #     z_0 = self.sample_gaussian(shape=(batch_size, self.generator.z_d_dim)).to(Y.device)
        # else:
        #     z_0 = z_0.to(Y.device)

        z_0 = self.sample_gaussian(shape=(batch_size, self.generator.z_d_dim)).to(Y.device)
        print("add z_0",torch.cuda.memory_allocated())
        # Sample z
        z = self.infer_z(z=z_0, Y=Y, mask=mask, n_iters=self.z_iters, with_noise=self.z_with_noise)
        
        # Generator
        Y_hat = self.generator(z)

        if self.normalize_windows:
            Y = Y*std + mean
            Y_hat = Y_hat*std + mean

        # if (self.z_persistent) and (idxs is not None):
        #     if self.univariate:
        #         z = z.reshape(initial_batch_size, n_features, self.generator.z_d_dim)
        #         self.p_0_chains[idxs] = z.to(self.p_0_chains.device)
        #     else:
        #         self.p_0_chains[idxs] = z.to(self.p_0_chains.device)

        return Y, Y_hat, z

