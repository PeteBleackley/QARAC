#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 08:08:03 2024

@author: peter
"""

import torch

class FactorizedMatrixMultiplication(torch.nn.Module):
    
    def __init__(self,size):
        super(FactorizedMatrixMultiplication,self).__init__()
        self.left = torch.nn.parameter.Parameter(torch.empty((size,8)))
        self.right = torch.nn.parameter.Parameter(torch.empty((8,size)))
        sigma = (3.0/(4.0*size))**0.25
        torch.nn.init.normal_(self.left,0.0,sigma)
        torch.nn.init.normal_(self.right,0.0,sigma)
        self.matrix = torch.tensordot(self.left,self.right,1)
        
    def forward(self,X):
        print(X.device)
        print(self.matrix.device)
        return torch.einsum('ij,klj->kli',self.matrix,X)