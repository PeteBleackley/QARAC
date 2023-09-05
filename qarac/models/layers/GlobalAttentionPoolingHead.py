#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 07:32:55 2023

@author: peter
"""

import keras
import tensorflow

class GlobalAttentionPoolingHead(keras.layers.Layer):
    
    def __init__(self):
        super(GlobalAttentionPoolingHead,self).__init__()
        self.global_projection = None
        self.local_projection = None
        
        
    def build(self,input_shape):
        width = input_shape[-1]
        self.global_projection = self.add_weight('global projection',shape=(width,width))
        self.local_projection = self.add_weight('local projection',shape=(width,width))
        self.build=True
    
    @tensorflow.function
    def project(self,X):
        return tensorflow.tensordot(X,self.local_projection,axes=1)
    
    def attention_function(self,gp):
        @tensorflow.function
        def inner(lp):
            return tensorflow.tensordot(lp,gp,axes=1)
        return inner
        
    def call(self,X,training=None):
        gp = tensorflow.linalg.l2_normalize(tensorflow.tensordot([tensorflow.reduce_sum(X,
                                                                                       axis=1),
                                                                  self.global_projection],
                                                                 axes=1),
                                            axis=1)
        lp = tensorflow.linalg.l2_normalize(tensorflow.ragged.map_flat_values(self.project,
                                                                              X),
                                            axis=2)
        attention = tensorflow.ragged.map_flat_values(self.attention_function(gp), 
                                                      lp)
        return tensorflow.reduce_sum(attention *X,
                                     axis=1)