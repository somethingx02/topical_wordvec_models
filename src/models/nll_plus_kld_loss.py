# -*- coding: utf8 -*-

import torch

# torch.nn and torch two different module from torch
import torch.nn
import torch.nn.functional
from torch.distributions.normal import Normal

import torch.autograd

if __name__ == '__main__':

    # x = torch.autograd.Variable(torch.Tensor([1, 1]))
    # p = torch.nn.Parameter(torch.Tensor(2))
    # a = x * p
    # b = a + 3
    # c = b * b * 3
    # out = c.mean()
    # out.backward()
    # print(out.data[0])
    # print(p.data)
    # print(p.grad.data)
