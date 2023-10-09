#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 07:32:55 2023

@author: peter
"""

import torch

EPSILON = 1.0e-12   

class GlobalAttentionPoolingHead(torch.nn.Module):
    
    def __init__(self,config):
        """
        Creates the layer
        Parameters
        ----------
        config : transformers.RobertaConfig
                 the configuration of the model

        Returns
        -------
        None.

        """
        size = config.hidden_size
        super(GlobalAttentionPoolingHead,self).__init__()
        self.global_projection = torch.nn.Linear(size,size,bias=False)
        self.local_projection = torch.nn.Linear(size,size,bias=False)
        
    
        
    def forward(self,X,attention_mask=None):
        """
        

        Parameters
        ----------
        X : torch.Tensor
            Base model vectors to apply pooling to.
        attention_mask: tensorflow.Tensor, optional
            mask for pad values
        

        Returns
        -------
        torch.Tensor
            The pooled value.

        """
        if attention_mask is None:
            attention_mask = torch.ones_like(X)
        else:
            attention_mask = attention_mask.unsqueeze(2)
        Xa = X*attention_mask
        sigma = torch.sum(Xa,dim=1)
        psigma = self.global_projection(sigma)
        nsigma = torch.max(torch.linalg.vector_norm(psigma,dim=1),EPSILON)
        gp = psigma/nsigma
        loc = self.local_projection(Xa)
        nloc = torch.max(torch.linalg.vector_norm(loc,dim=2),EPSILON)
        lp = loc/nloc
        attention = torch.einsum('ijk,k->ij',lp,gp)
        return torch.einsum('ij,ijk->ik',attention,Xa)