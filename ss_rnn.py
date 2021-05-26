# encoding: utf-8
'''
@author: yang shu
@contact: yangshuinhust@gmail.com
@software: vscode
@file: ss_rnn.py
@time: 2021-05-25
@desc: 神经网络定义
'''

import numpy as np
import torch
from torch import nn

dtype = torch.float

# the simplest RNN, which is the SS
class Ss_rnn(nn.Module):
    # 状态空间模型等价为有两个线性层的RNN
    def __init__(self,
                 input_size,
                 state_size,
                 output_size,
                 u_mean,
                 u_std,
                 s_mean,
                 s_std,
                 step_size,
                 bias=True):

        super(Ss_rnn, self).__init__()
        # the network info
        self.input_size = input_size
        self.state_size = state_size
        self.output_size = output_size

        # the scaling info for normalization
        self.u_mean = u_mean
        self.u_std = u_std
        self.s_mean = s_mean
        self.s_std = s_std

        # other info book keeping the model property
        self.step_size = step_size

        # add the input to hidden layer
        # input to hidden1--------------------------------------------------------------
        # input: manipulated input  u(t)
        #        state              s(t-1),
        # output: state             s(1)
        self.u2x = nn.Linear(input_size + state_size, state_size, bias=bias)
        # the output layers-------------------------------------------------------------
        # hidden(i-1) to hidden(i)
        # input:
        #       state              s(t-1),
        #       hidden             h(i-1)
        self.x2y = nn.Linear(state_size, output_size, bias=bias)

    def forward(self, input_seq, state_init=None):
        
        dtype = torch.float
        # the dim info of input
        series_length = input_seq.size(0)
        # init hidden states
        if state_init is None:
            state_init = torch.zeros(self.state_size)
        state_new = state_init
        # init the output
        series_output = torch.zeros(series_length,
                                   dtype=dtype)
        # start training
        for i in range(input_seq.size(0)):
            # iterate through time
            input_u = input_seq[i, :]
            state_new,output_new = self.forward_step(input_u=input_u, state_input=state_new)
            # to the output tensor
            series_output[i] = output_new
        return series_output

    def forward_step(self, input_u, state_input):
        # propagate one step in time
        # input to hidden-----------------------------------------------------
        input_combined = torch.cat((input_u, state_input), 0)
        state_new = self.u2x(input_combined)
        output_new=self.x2y(state_new)

        return state_new,output_new