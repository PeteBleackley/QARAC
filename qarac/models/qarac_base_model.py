#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 09:50:14 2023

@author: peter
"""

import keras
import qarac.models.layers.HierarchicalLogits
import qarac.models.layers.HyenaLayer

def qarac_base_model(vocab_size,width,depth,decoder=True):
    print('Building','decoder' if decoder else 'encoder','model with vocab size',
          vocab_size,',',depth,'layers and vector width',width)
    stack = [keras.layers.Input(shape=(None,),ragged=True),
             keras.layers.Embedding(vocab_size,width,name='Embedding')]
    for _ in range(depth):
        stack.append(qarac.models.layers.HyenaLayer.HyenaLayer(causal=decoder))
    #stack.append(keras.layers.TimeDistributed(qarac.models.layers.HierarchicalLogits.HierarchicalLogits(vocab_size)))
    #stack.append(keras.layers.TimeDistributed(keras.layers.Softmax()))
    stack.append(keras.layers.TimeDistributed(keras.layers.Dense(vocab_size,activation='softmax')))
    return keras.models.Sequential(stack)