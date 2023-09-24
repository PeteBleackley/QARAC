#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 07:32:55 2023

@author: peter
"""

import keras
import tensorflow

@tensorflow.function
def dot_prod(x,y):
    return tensorflow.tensordot(x,y,axes=1)

class GlobalAttentionPoolingHead(keras.layers.Layer):
    
    def __init__(self):
        """
        Creates the layer

        Returns
        -------
        None.

        """
        super(GlobalAttentionPoolingHead,self).__init__()
        self.global_projection = None
        self.local_projection = None
        
        
    def build(self,input_shape):
        """
        Initialises layer weights

        Parameters
        ----------
        input_shape : tuple
            Shape of the input layer

        Returns
        -------
        None.

        """
        width = input_shape[-1]
        self.global_projection = self.add_weight('global projection',shape=(width,width))
        self.local_projection = self.add_weight('local projection',shape=(width,width))
        self.built=True
    
    @tensorflow.function
    def project_local(self,X):
        return tensorflow.tensordot(X,
                                    self.local_projection,
                                    axes=1)
        
    def call(self,X,training=None):
        """
        

        Parameters
        ----------
        X : tensorflow.Tensor
            Base model vectors to apply pooling to.
        training : bool, optional
            Not used. The default is None.

        Returns
        -------
        tensorflow.Tensor
            The pooled value.

        """
        gp = tensorflow.linalg.l2_normalize(tensorflow.tensordot(tensorflow.reduce_sum(X,
                                                                                       axis=1),
                                                                  self.global_projection,
                                                                 axes=1),
                                            axis=1)
        lp = tensorflow.linalg.l2_normalize(tensorflow.vectorized_map(self.project_local,
                                                                      X),
                                            axis=2)
        attention = tensorflow.tensordot(lp,gp,axes=1)
        return tensorflow.reduce_sum(attention *X,
                                     axis=1)