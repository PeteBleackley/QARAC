#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:01:39 2023

@author: peter
"""

import qarac.models.QaracBaseModel
import qarac.models.layers.GlobalAttentionPoolingHead

class QaracEncoderModel(qarac.models.QaracBaseModel.QaracBaseModel):
    
    def __init__(self,base_model):
        """
        Creates the endocer model

        Parameters
        ----------
        base_model : transformers.TFRobertaModel
            The base model

        Returns
        -------
        None.

        """
        super(QaracEncoderModel,self).from_pretrained(base_model)
        self.head = qarac.models.layers.GlobalAttentionPoolingHead.GlobalAttentionPoolingHead(self.base_model.config)
        
        
    def forward(self,input_ids,
             attention_mask=None):
        """
        Vectorizes a tokenised text

        Parameters
        ----------
        inputs : tensorflow.Tensor
            tokenized text to endode

        Returns
        -------
        tensorflow.Tensor
            Vector representing the document

        """

        return self.head(self.base_model(input_ids,
                                         attention_mask).last_hidden_state,
                         attention_mask)
    
    @property
    def config(self):
        return self.base_model.config
  
    
    