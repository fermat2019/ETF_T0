from tool import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class ManualGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ManualGRU, self).__init__()
        self.hidden_size = hidden_size

        # 输入到隐藏层的权重
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        # 隐藏层到隐藏层的权重
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        # 偏置项
        self.bias_ih = nn.Parameter(torch.randn(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(3 * hidden_size))

    def forward(self, input, h_prev):
        """
        input: (batch_size, input_size)
        h_prev: (batch_size, hidden_size)
        """
        # 计算输入和隐藏状态的线性变换
        gates_ih = torch.mm(input, self.weight_ih.t()) + self.bias_ih
        gates_hh = torch.mm(h_prev, self.weight_hh.t()) + self.bias_hh

        # 将线性变换的结果拆分为重置门、更新门和候选隐藏状态
        r_gate, z_gate, n_gate = gates_ih.chunk(3, 1)
        r_gate_h, z_gate_h, n_gate_h = gates_hh.chunk(3, 1)

        # 重置门和更新门的激活函数
        r_gate = torch.sigmoid(r_gate + r_gate_h)
        z_gate = torch.sigmoid(z_gate + z_gate_h)

        # 候选隐藏状态
        n_gate = torch.tanh(n_gate + r_gate * n_gate_h)

        # 更新隐藏状态
        h_next = (1 - z_gate) * n_gate + z_gate * h_prev

        return h_next