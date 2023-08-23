#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 09:50:14 2023

@author: peter
"""

import keras
import layers

def quarac_base_model(vocab_size,width,depth,decoder=True):
    stack = [keras.layers.Embedding(vocab_size,width)]
    for _ in range(depth):
        stack.append(layers.HyenaLayer(causal=decoder))
    stack.append(keras.layers.Timedistributed(layers.HierarchicalLogits()))
    stack.append(keras.layers.Timedistributed(keras.layers.Softmax()))
    return keras.models.Sequential(stack)