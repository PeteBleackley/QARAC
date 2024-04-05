#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:01:39 2023

@author: peter
"""

import transformers
import qarac.models.layers.GlobalAttentionPoolingHead

class QaracEncoderModel(transformers.PreTrainedModel):
    
    def __init__(self,path):
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
        config = transformers.PretrainedConfig.from_pretrained(path)
        super(QaracEncoderModel,self).__init__(config)
        self.encoder = transformers.RobertaModel.from_pretrained(path)
        self.head = qarac.models.layers.GlobalAttentionPoolingHead.GlobalAttentionPoolingHead(config)
        
        
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
        print('Encoder',self.encoder.device)
        print('Head',self.head.device)
        if attention_mask is None and 'attention_mask' in input_ids:
            (input_ids,attention_mask) = (input_ids['input_ids'],input_ids['attention_mask'])
        print('input_ids',input_ids.device)
        print('attention_mask',attention_mask.device)
        return self.head(self.encoder(input_ids,
                                      attention_mask).last_hidden_state,
                         attention_mask)
    
  
    
    