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


@tensorflow.function
def convolve(x,y):
    xT = tensorflow.vectorized_map(tensorflow.transpose, x)
    yT = tensorflow.vectorized_map(tensorflow.transpose, y)
    fx = tensorflow.vectorized_map(tensorflow.signal.rfft, xT)
    fy = tensorflow.vectorized_map(tensorflow.signal.rfft, yT)
    fz = fx*fy
    zT = tensorflow.vectorized_map(tensorflow.signal.irfft, fz)
    return tensorflow.vectorized_map(tensorflow.transpose,zT)

# @tensorflow.function    
# def fft(x):
#     return tensorflow.signal.rfft(tensorflow.transpose(x))
 
# @tensorflow.function   
# def ifft(x):
#     return tensorflow.transpose(tensorflow.signal.irfft(x))

@tensorflow.function
def pad(x):
    return tensorflow.concat([x,tensorflow.zeros_like(x)],0)

@tensorflow.function()
def truncate(args):
    (data,length)=args
    return data[:length]

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
    
    @tensorflow.function
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
        self.built = True
        
    def conpute_output_shape(self,input_shape):
        return input_shape
        
    @tensorflow.function
    def project(self,x):
        return tensorflow.tensordot(x,self.data_projection,axes=1)
    
    @tensorflow.function
    def generate_filters(self,t):
        return tensorflow.tensordot(t, self.filters,axes=1)
        
    def call(self,X,training=None):
        
        x = tensorflow.ragged.map_flat_values(self.project, X)
        f = tensorflow.ragged.map_flat_values(self.generate_filters,self.positional_encoding(X))
        if self.causal:
            x = tensorflow.vectorize_map(pad,x)
            f = tensorflow.vectorize_map(pad,f)
        y = x[:,:,:,0]
        for i in tensorflow.range(self.stages):
            y = convolve(y,f[:,:,:,i])*x[:,:,:,i+1]
        if self.causal:
            y = tensorflow.vectorized_map(truncate,(y,X.row_lengths()))
        return tensorflow.raw_ops.RaggedTensorToVariant(rt_nested_splits=y.row_splits,
                                                        rt_dense_values=y.flat_values,
                                                        batched_input=True)