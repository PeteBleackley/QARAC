#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:34:14 2023

@author: peter
"""

import keras
import keras_nlp
import tensorflow

def convolve(x,y):
    xT = tensorflow.transpose(x,[0,2,1])
    yT = tensorflow.transpose(y,[0,2,1])
    z = tensorflow.signal.irfft(tensorflow.signal.rfft(xT)*tensorflow.signal.rfft(yT))
    return tensorflow.transpose(z,[0,2,1])

    
    

class HyenaLayer(keras.layers.Layer):
    """Keras implementation of Hyena layer. Unlike in the original paper,
       this can be used as an encoder layer by setting the optional parameter
       `causal` to `False`"""
       
    def __init__(self,stages=3,causal=True):
        """
        

        Parameters
        ----------
        stages : int, optional
            Number of stages of convolution and Hadamard multiplication. The default is 3.
        causal : bool, optional
            Set to False for an encoder layer. The default is True.

        Returns
        -------
        None.

        """
        
        self.stages = stages
        self.causal = causal
        self.data_projection = None
        self.filters = None
        self.positional_encoding = keras_nlp.layers.SinePositionalEmbedding()
        
    def build(self,input_shape):
        self.data_projection = keras.layers.TimeDistributed(keras.layers.Dense((self.stages+1,input_shape[2]),
                                                                               activation='linear'))
        self.filters = keras.layers.TimeDistributed((self.stages,input_shape[2]),
                                                    activation='linear')
        
    def call(self,X,training=None):
        x = self.data_projection(X)
        f = self.filters(self.positional_encoding(X))
        if self.causal:
            concat = keras.layers.Concatenate()
            x = concat(x,tensorflow.zeros_like(x))
            f = concat(f,tensorflow.zeros_like(f))
        y = x[0]
        for i in range(self.stages):
            y = convolve(y,f[i])*x[i+1]
        if self.causal:
            for (i,n) in enumerate(X.row_lengths()):
                y[i] = y[i,:n]
        return y