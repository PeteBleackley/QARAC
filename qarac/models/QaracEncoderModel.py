#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:01:39 2023

@author: peter
"""

import transformers
import qarac.models.layers.GlobalAttentionPoolingHead

class QaracEncoderModel(transformers.TFPreTrainedModel):
    
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
        super(QaracEncoderModel,self).__init__(base_model.config)
        self.base_model = base_model
        self.head = qarac.models.layers.GlobalAttentionPoolingHead.GlobalAttentionPoolingHead()
        
    def build(self,input_shape):
        """
        

        Parameters
        ----------
        input_shape : tuple
            shape of input data.

        Returns
        -------
        None.

        """
        self.built=True
        
    def call(self,inputs):
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

        return self.head(self.base_model(inputs).last_hidden_state)
  
    
    