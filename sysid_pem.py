# encoding: utf-8
'''
@author: yang shu
@contact: yangshuinhust@gmail.com
@software: vscode
@file: ss_rnn.py
@time: 2021-05-25
@desc: 状态空间模型识别的主要流程
'''
import random
from os import path

from ss_rnn import Ss_rnn
from data_preprocessing import Data_preprocess
import matplotlib.pyplot as plt
import numpy as np

import torch

from torch import nn

dtype = torch.float
# bs is batch size; u_dim is input dimension;


class Sys_id_pem(object):
    
    def __init__(self,process_data):
        self.process_data=process_data
        
    def fit(self,step_size,mdl_path,num_state=4,epochs=100,lr=1e-3,l2_pen=1e-3,subseries_len=200,plot_results=True):
    # 根据数据拟合一个状态空间模型
        ss_mdl,loss_train, loss_val = train_model(
                                       epochs=epochs,
                                       train_x=self.process_data.x,
                                       train_y=self.process_data.y, 
                                       num_state=num_state,
                                       lr=lr,
                                       l2_pen=l2_pen, 
                                       subseries_len=subseries_len, 
                                       u_mean=self.process_data.x_mean, 
                                       u_std=self.process_data.x_var, 
                                       y_mean=self.process_data.y_mean, 
                                       y_std=self.process_data.y_var,
                                       step_size=step_size,
                                       model_path=mdl_path)
        if plot_results:
            plt.figure(1)
            plt.plot(loss_train, label="training loss")
            plt.plot(loss_val, label="validation loss")
            plt.legend()


            u,y_pred,y_true=plot_val( self.process_data.x[3],self.process_data.y[3],ss_mdl)
            fig, (ax1, ax2) = plt.subplots(2)
            fig.suptitle('validation plot')
            ax1.plot(y_pred, label="predicted ouput")
            ax1.plot(y_true, label="actual ouput")
            ax1.legend()
            ax2.plot(u)
            plt.show()
        
        self.mdl=ss_mdl
        return ss_mdl
    
    def get_ss_mat(self,ss_mdl_path): 
    # 输出主要模型参数   
        A,B,C=get_ss(self.mdl)
        np.savez(ss_mdl_path, A=A, B=B,C=C)
        return A,B,C
    
def train_model(epochs, train_x,train_y, num_state, lr, l2_pen, subseries_len,  u_mean, u_std, y_mean, y_std,
                step_size, model_path,print_every=10):
    # 模型训练的主要过程
    loss_train = np.zeros(epochs)
    loss_val = np.zeros(epochs)
    loss_temp = 0

    best_loss = 10  # 初始化模型loss
    # 如果没有训练好的模型，则重新开始训练一个
    if path.exists(model_path):
        ss_model = torch.load(model_path)
    else:
        ss_model = Ss_rnn(  input_size=3,
                            state_size=num_state,
                            output_size=1,
                            u_mean=u_mean,
                            u_std=u_std,
                            s_mean=y_mean,
                            s_std=y_std,
                            step_size=step_size,
                            bias=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ss_model.parameters(),
                                 lr=lr,
                                 weight_decay=l2_pen)
    # 开始训练--------------------------------------------
    for epoch in range(epochs):
        ss_model.train()

        loss = 0
        # 在数据中随机选择一小段数据
        train_rand_index = random.randint(0, 2)
        ub=train_x[train_rand_index]
        sb=train_y[train_rand_index]
        series_length = sb.size(0)
        itr_per_epoch=int(series_length/subseries_len)
        
        for itr in range(itr_per_epoch):
            start_index=random.randint(0, series_length-subseries_len-10)
            ub_sub=ub[start_index:start_index+subseries_len,:]
            sb_sub=sb[start_index:start_index+subseries_len]
            
            # 先前传递
            state_output = ss_model(input_seq=ub_sub)
            # 计算loss
            loss = criterion(state_output, sb_sub)
            # 反向传递
            loss.backward()
            optimizer.step()
            
            optimizer.zero_grad()
            loss_temp += (loss).item() /itr_per_epoch
        # validation现有模型
        ss_model.eval()  # ----------------------the validation
        with torch.no_grad():
            #第四段序列被选择为validation dataset
            ub=train_x[3]
            sb=train_y[3]
            state_output = ss_model(input_seq=ub)
            valid_loss = criterion(state_output, sb)
            valid_loss = valid_loss.item()
        # 计算loss
        loss_train[epoch] = loss_temp
        loss_val[epoch] = valid_loss
        
        if epoch % print_every == 1:
            print("the training loss of epoch", epoch, " is:", loss_temp)
            print("the validation loss of epoch", epoch, " is:", valid_loss)
        loss_temp = 0
        # 如果再validation数据集上有进步，则储存模型
        if best_loss > valid_loss:
            print('save model at epoch ', epoch, ' with validation loss ',
                  valid_loss)
            best_loss = valid_loss
            torch.save(ss_model, model_path)
    return ss_model,loss_train, loss_val

def plot_val( val_x,val_y,mdl):
    # 显示模型预测的序列
    mdl.eval() 
    with torch.no_grad():
        y_pred = mdl(input_seq=val_x)
        
    y_pred=y_pred.detach().numpy()
    y_true=val_y.detach().numpy()
    u=val_x.detach().numpy()
    
    return u,y_pred,y_true


def get_ss(mdl):
    # 输出学习得到的模型参数
    state_trans=mdl.u2x.weight.detach().numpy()
    C=mdl.x2y.weight.detach().numpy()
    
    num_state=state_trans.shape[0]
    
    A=state_trans[:,0:num_state]
    B=state_trans[:,-3:]   

    return A,B,C 

# start
if __name__ == "__main__":
    data_path='raw_data.csv'
    process_data=Data_preprocess(data_path)
    sysid_ss_pem=Sys_id_pem(process_data=process_data)

    step_size = 5.0 / 60  #5 秒

    ss_mdl=sysid_ss_pem.fit(epochs=1,step_size=step_size,mdl_path='model/saved_model.tar')
    
    A,B,C=sysid_ss_pem.get_ss_mat()
    
    
    print(A)
    print(B)
    print(C)
    