#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 15:22:21 2021

@author: peter
"""

import keras
import tensorflow

class LeafNode(keras.layers.Layer):
    def __init__(self):
        self.bias = self.add_weight(shape=(1,),
                                    initializer='random_normal',
                                    trainable=True)
        
    def build(self,input_shape):
        pass
    
    def call(self,X,training=None):
        return self.bias

class HierarchicalLogits(keras.layers.Layer):
    
    def __init__(self,n):#structure,row=-1,order=None):
        super(HierarchicalLogits,self).__init__()
        # self.structure = structure
        # self.row = row
        self.normal = None
        
        self.n_outputs = n
        l = n//2
        if l==1:
            self.left=LeafNode()
        else:
            self.left=HierarchicalLogits(l)
        if n-l==1:
            self.right=LeafNode()
        else:
            self.right=HierarchicalLogits(n-l)
        self.concat = keras.layers.Concatenate()
            
        
    def build(self,input_shape):
        self.normal = self.add_weight(shape=(input_shape[-1],),
                                       initializer='random_normal',
                                       trainable=True)
        self.left.build(input_shape)
        self.right.build(input_shape)
        
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]+(self.n_outputs,)
        
    def call(self,X,training=None):
        
        y=tensorflow.tensordot(X,self.normal,1)
        result = self.concat([self.left(X)+y,self.right(X)]-y)
        return result
        
    
    def get_config(self):
        return {'n':self.n_outputs}
    
    @classmethod
    def from_config(cls,config):
        return cls(config['n'])
        
    