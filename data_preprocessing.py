# encoding: utf-8
'''
@author: yang shu
@contact: yangshuinhust@gmail.com
@software: vscode
@file: data_preprocessing.py
@time: 2021-05-25
@desc: 数据处理
'''
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

class Data_preprocess(object):
    def __init__(self,data_path):

        
        raw_data=(pd.read_csv(data_path)).to_numpy()
        x_lin,y_lin=self.cal_linear_feature(raw_data)   # 把每个列重新组合成能量守恒的形式
        x_exp,y_exp=self.split_exp(x_lin,y_lin)         # 把数据分成几段稳态的数据
        x_norm,y_norm,x_mean,x_var,y_mean,y_var=self.normalize_data(x_exp,y_exp)   # 数据标准化
    
        
        self.x=x_norm
        self.y=y_norm
        self.x_mean=x_mean
        self.y_mean=y_mean
        self.x_var=x_var
        self.y_var=y_var

    def cal_linear_feature(self,raw_data):
        # 把每个列重新组合成能量守恒的形式， 包含 
        #   冷却水制冷量
        #   炉膛对流换热
        #   风门开度
        q_cool=(raw_data[:,5]-raw_data[:,2])*raw_data[:,1]
        q_chamber_conv=(raw_data[:,4]-raw_data[:,2])
        gate_open=raw_data[:,6]
        
        x=np.column_stack((q_cool,gate_open,q_chamber_conv))
        y=(raw_data[:,7]-raw_data[:,2])
        
        return x,y
        
    def split_exp(self,x,y):  
        # 把数据分成几列稳态数据  
        exp1=range(4300,18800);
        exp2=range(20300,24500);
        exp3=range(26000,39000);
        exp4=range(40010,46600);
        exp5=range(48000,59900);
        
        x_slice=[x[exp1,:],x[exp2,:],x[exp3,:],x[exp4,:],x[exp5,:]]
        y_slice=[y[exp1],y[exp2],y[exp3],y[exp4],y[exp5]]
        
        return x_slice,y_slice
        
    def normalize_data(self,x,y):
        # 数据标准化
        num_exp=len(x)
        
        x_all=x[0]
        y_all=y[0]
        for i in range(1,num_exp):
            x_all=np.concatenate((x_all,x[i]))
            y_all=np.concatenate((y_all,y[i]))
        
        x_mean=np.mean(x_all,axis=0)
        x_var=np.sqrt(np.var(x_all,axis=0))
        
        y_mean=np.mean(y_all,axis=0)
        y_var=np.sqrt(np.var(y_all,axis=0))
        
        for i in range(num_exp):
            x[i]=(x[i]-x_mean)/x_var
            y[i]=(y[i]-y_mean)/y_var
            # add dim for RNN training
            
            x[i]=torch.tensor(x[i]).float()
            y[i]=torch.tensor(y[i]).float()
            
        return x,y,x_mean,x_var,y_mean,y_var
            
    
if __name__ == "__main__":
    file_path='raw_data.csv'
    dataraw=Data_preprocess(file_path)
