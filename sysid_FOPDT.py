# encoding: utf-8
'''
@author: yang shu
@contact: yangshuinhust@gmail.com
@software: vscode
@file: sysid_FOPDT.py
@time: 2021-05-25
@desc: 计算 k t tau
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from data_preprocessing import Data_preprocess

# 通过最小化预测误差，来拟合一个 一阶惯性加纯滞后 (FOPDT) 模型
# 如果需要速度的 K T tau， 则对y求一次一阶差分带入即可

data_path='raw_data.csv'

step_size = 5.0 / 60  # 5 秒
process_data=Data_preprocess(data_path)

# 选择一小段数据来拟合 （FOPDT是很简单的模型，不需要很大的数据量，同时次拟合方法处理大数据较慢）
u=process_data.x[1].detach().numpy()[1250:1500,:]   
yp=process_data.y[1].detach().numpy()[1250:1500]

num_step=yp.shape[0]
t=step_size*np.arange(num_step)

u0 = u[0,:]
yp0 = yp[0]

delta_t = t[1]-t[0]
# FOPDF 为连续时间模型，所以对数据进行内插
uf=[]
for i in range(3):
    uf.append(interp1d(t,u[:,i]))


# 定义FOPDT dy/dt= f(...)
def fopdt(y,t,uf,Km,taum,thetam):
    # 参数
    #  y      = 输出
    #  t      = 时间
    #  uf     = 内插的u
    #  Km     = 模型增益 （3维向量）
    #  taum   = 模型时间常数（3维向量）
    #  thetam = 模型时延（3维向量）
    
    dydt=0
    for i in range(3):
        try:
            if (t-thetam[i]) <= 0:
                um = uf[i](0.0)
            else:
                um = uf[i](t-thetam[i])
        except:
            um = u0[i]
        dydt = dydt+(-(y-yp0) + Km[i] * (um-u0[i]))/taum[i]
    return dydt.item()

# 计算输出
def sim_model(mdl_para):
    # 模型参数
    Km = mdl_para[0:3]  # 模型增益
    taum = mdl_para[3:6]    # 模型时间常数
    thetam = mdl_para[6:9]  # 模型时延
    
    # 初始化
    ym = np.zeros(num_step)  
    ym[0] = yp0
    
    # 使用积分计算输出  
    for i in range(0,num_step-1):
        ts = [t[i],t[i+1]]
        y1 = odeint(fopdt,ym[i],ts,args=(uf,Km,taum,thetam))
        ym[i+1] = y1[-1]
    return ym

# 定义MSE误差（我们希望最小化的）
def objective(mdl_para):
    ym = sim_model(mdl_para)
    obj = 0.0
    for i in range(len(ym)):
        obj = obj + (ym[i]-yp[i])**2 # 计算MSE
    return obj


# 初步猜测
mdl_para_init = np.zeros(9)
mdl_para_init[0] = 0.3 # u1 模型增益 
mdl_para_init[1] = -0.3 # u2 模型增益 
mdl_para_init[2] =-0.3 # u3 模型增益 
mdl_para_init[3] = 0.3 # u1 模型时间常数 1
mdl_para_init[4] = 0.3 # u2 模型时间常数 2
mdl_para_init[5] =0.3 # u3 模型时间常数 3
mdl_para_init[6] = 0.0 # u1 模型时延 
mdl_para_init[7] = 0.0 # u2 模型时延 
mdl_para_init[8] =0.0 # u3 模型时延 


# 计算初始猜测的结果
print('Initial model MSE: ' + str(objective(mdl_para_init)))

#  约束求解范围 （共9个区间，对应9个初步猜测）
bnds = ((0.0, 1.0),(-1.0, 1.0),(-1.0, 1.0), (0.1,1.0),(0.1,1.0),(0.1,1.0), (0.0, 1.0),(0.0, 1.0),(0.0, 1.0))
solution = minimize(objective,mdl_para_init,bounds=bnds,method='SLSQP')
mdl_para = solution.x

# 计算优化后的结果
print('Final model MSE: ' + str(objective(mdl_para)))

# 拟合的 k t tau
print('Kp1: ' + str(mdl_para[0]))
print('Kp2: ' + str(mdl_para[1]))
print('Kp3: ' + str(mdl_para[2]))

print('tau1: ' + str(mdl_para[3]))
print('tau2: ' + str(mdl_para[4]))
print('tau3: ' + str(mdl_para[5]))

print('theta1: ' + str(mdl_para[6]))
print('theta2: ' + str(mdl_para[7]))
print('theta3: ' + str(mdl_para[8]))

# 对面初步猜测和优化后的拟合结果
ym1 = sim_model(mdl_para_init)
ym2 = sim_model(mdl_para)
# 画图
plt.figure()
plt.subplot(4,1,1)
plt.plot(t,yp,'kx-',linewidth=2,label='Process Data')
plt.plot(t,ym1,'b-',linewidth=2,label='Initial Guess')
plt.plot(t,ym2,'r--',linewidth=3,label='Optimized FOPDT')
plt.ylabel('Output')
plt.legend(loc='best')
plt.subplot(4,1,2)
plt.plot(t,u[:,0],'bx-',linewidth=2)
plt.plot(t,uf[0](t),'r--',linewidth=3)
plt.legend(['Measured','Interpolated'],loc='best')
plt.ylabel('u1')
plt.subplot(4,1,3)
plt.plot(t,u[:,1],'bx-',linewidth=2)
plt.plot(t,uf[1](t),'r--',linewidth=3)
plt.legend(['Measured','Interpolated'],loc='best')
plt.ylabel('u2')
plt.subplot(4,1,4)
plt.plot(t,u[:,2],'bx-',linewidth=2)
plt.plot(t,uf[2](t),'r--',linewidth=3)
plt.legend(['Measured','Interpolated'],loc='best')
plt.ylabel('u3')
plt.show()