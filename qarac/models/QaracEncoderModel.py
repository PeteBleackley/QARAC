#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:01:39 2023

@author: peter
"""

import transformers
import qarac.layers.GlobalAttentionPoolingHead

class QaracEncoderModel(transformers.TFPretrainedModel):
    
    def __init__(self,base_model):
        super(QaracEncoderModel,self).__init__()
        self.base_model = base_model
        self.head = qarac.layers.GlobalAttentionPoolingHead.GlobalAttentionPoolingHead()
        
    def call(self,inputs):
        return self.head(self.base_model(inputs).last_hidden_state)
  
    
    