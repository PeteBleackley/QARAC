#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:52:39 2023

@author: peter
"""

import transformers

class QaracBaseModel(transformers.PreTrainedModel):
    """Base class for Qarac Models. Provided config_class"""
    config_class = transformers.PretrainedConfig
    
    def __init__(self,config,*inputs,**kwargs):
        super(QaracBaseModel,self).__init__(config,*inputs,**kwargs)