from abc import ABC,abstractmethod
from statistics import mode
from turtle import forward
from typing import OrderedDict
import torch
from model import model
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from base import Base
import torch.nn.functional as F

class KeyPoint(Base):
    def __init__(self):
        super(KeyPoint,self).__init__()
        self.netKey = model(15).double()
        self.opt = Adam(self.netKey.parameters(),lr=0.001)
        self.optimizers.append(self.opt)
        self.schedulers.append(StepLR(self.opt,step_size=10, gamma=0.3))        
        self.network_names =['Key']
        self.loss_names= ['mce']

    def forward(self):
        self.output = self.netKey(self.input)

    def backward(self):
        self.loss_mce = F.mse_loss(self.output,self.label)

    def optimize_parameters(self):
        self.loss_mce.backward()
        self.opt.step()
        self.opt.zero_grad()
       
