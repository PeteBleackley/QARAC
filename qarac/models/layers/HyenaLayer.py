#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 09:34:14 2023

@author: peter
"""

import keras
import keras_nlp
import tensorflow
import warnings

def convolve(x,y):
    
    fx = tensorflow.vectorized_map(fft, x, warn=False)
    fy = tensorflow.vectorized_map(fft, y, warn=False)
    fz = fx*fy
    return tensorflow.vectorized_map(ifft,fz,warn=False)

@tensorflow.function    
def fft(x):
    return tensorflow.signal.rfft(tensorflow.transpose(x))
 
@tensorflow.function   
def ifft(x):
    return tensorflow.transpose(tensorflow.signal.irfft(x))


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
        super(HyenaLayer,self).__init__()
        self.stages = stages
        self.causal = causal
        self.data_projection = None
        self.filters = None
    
    def positional_encoding(self,X):
        t = tensorflow.dtypes.saturate_cast(tensorflow.ragged.range(X.row_lengths()),
                                            tensorflow.float32)
        width = X.shape[-1]//2
        f =10000 **tensorflow.expand_dims(-tensorflow.range(width,
                                                            dtype=tensorflow.float32)/width,
                                           axis=0)
        phi = tensorflow.RaggedTensor.from_row_lengths(t.flat_values * f,
                                                       X.row_lengths())

        return tensorflow.concat([tensorflow.sin(phi),
                                  tensorflow.cos(phi)],
                                 axis=-1)
        
        
    def build(self,input_shape):
        width = input_shape[-1]
        self.data_projection = self.add_weight(shape=(width,width,self.stages+1),
                                               trainable=True)
        self.filters = self.add_weight(shape=(width,width,self.stages),
                                       trainable=True)
        
    def call(self,X,training=None):
        x_flat = tensorflow.tensordot(X.flat_values,
                                      self.data_projection,
                                      axes=1)
        f_flat = tensorflow.tensordot(self.positional_encoding(X).flat_values,
                                      self.filters,
                                      axes=1)
        x = tensorflow.RaggedTensor.from_row_lengths(x_flat,X.row_lengths())
        f = tensorflow.RaggedTensor.from_row_lengths(f_flat,X.row_lengths())
        if self.causal:
            concat = keras.layers.Concatenate()
            x = concat(x,tensorflow.zeros_like(x))
            f = concat(f,tensorflow.zeros_like(f))
        y = x[:,:,:,0]
        for i in range(self.stages):
            y = convolve(y,f[:,:,:,i])*x[:,:,:,i+1]
        if self.causal:
            for (i,n) in enumerate(X.row_lengths()):
                y[i] = y[i,:n]
        return y