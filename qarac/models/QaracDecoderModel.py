#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:29:03 2023

@author: peter
"""

import keras
import transformers

class QaracDecoderHead(keras.layers.Layer):
    
    def __init__(self,config):
        super(QaracDecoderHead,self).__init__()
        self.concat = keras.layers.Concatenate(axis=1)
        self.layer_0 = transformers.TFRobertaLayer(config)
        self.layer_1 = transformers.TFRobertalayer(config)
        self.head = transformers.TFRobertaLMHead(config)
        
    def call(self,inputs):
        vectors = self.concat(inputs)
        l0 = self.layer_0(vectors)
        return self.head(self.layer1(l0.last_hidden_state[:,1:]))

class QaracDecoderModel(transformers.TFPretrainedModel):
    
    def __init__(self,base_model):
        super(QaracDecoderModel,self).__init__()
        self.base_model = base_model
        self.decoder_head = QaracDecoderHead(self.base_model.config)
        
    def call(self,inputs):
        (v,s) = inputs
        return self.decoder_head((v,self.base_model(s)))
        
    