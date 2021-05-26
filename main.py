# encoding: utf-8
'''
@author: yang shu
@contact: yangshuinhust@gmail.com
@software: vscode
@file: ss_rnn.py
@time: 2021-05-25
@desc: 主函数
'''

from tokenize import String
from ss_rnn import Ss_rnn
from data_preprocessing import Data_preprocess
from sysid_pem import Sys_id_pem
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from os import path


    
if __name__ == "__main__":
    data_path='raw_data.csv'
    mdl_path='ss_mdl/ss_mat.npz'

    step_size = 5.0 / 60  #5 second
    process_data=Data_preprocess(data_path)
    sysid_ss_pem=Sys_id_pem(process_data=process_data)

    
    T=np.linspace(0.0, 3.0, num=1000)
    for num_state in range(1,11):
        print("try state number of "+str(num_state))
        ss_mdl_path='ss_mdl/ss_mat_'+str(num_state)+'state.npz'
        torch_mdl_path='model/saved_model_'+str(num_state)+'state.tar'
        # 训练以获得ABCD矩阵
        if path.exists(ss_mdl_path):
            print('model already exist')
            ss_mdl = np.load(ss_mdl_path)
            A=ss_mdl['A']
            B=ss_mdl['B']
            C=ss_mdl['C']
        else:
            print('train a new model')
            ss_mdl=sysid_ss_pem.fit(step_size=step_size,epochs=100,num_state=num_state,mdl_path=torch_mdl_path,plot_results=False)
            A,B,C=sysid_ss_pem.get_ss_mat(ss_mdl_path=ss_mdl_path)
            
        num_state=C.shape[1]
        num_input=B.shape[1]
        num_output=C.shape[0]
        
        # 绘制阶跃响应图
        sys_mdl=signal.StateSpace(A,B,C,np.zeros((num_output,num_input)),dt=step_size)
        t, y = signal.dstep(sys_mdl,t=T)
        fig, (ax1, ax2,ax3) = plt.subplots(3)
        fig.suptitle(str(num_state)+' state step response plot ')
        ax1.plot(t, np.squeeze(y[0]), label="heat from cooling")
        ax2.plot(t, np.squeeze(y[1]), label="gate position")
        ax3.plot(t, np.squeeze(y[2]), label="chamber temp")
        ax1.legend()
        ax2.legend()
        ax3.legend()
        
    plt.show()
    
    