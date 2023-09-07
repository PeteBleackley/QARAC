#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:29:03 2023

@author: peter
"""

import keras
import tensorflow
import transformers

class QaracDecoderHead(keras.layers.Layer):
    
    def __init__(self,config):
        """
        Creates the Decoder head

        Parameters
        ----------
        config : transformers.RobertaConfig
            Config for the RobertaModel that this head will be attached to.

        Returns
        -------
        None.

        """
        super(QaracDecoderHead,self).__init__()
        self.concat = keras.layers.Concatenate(axis=1)
        self.layer_0 = transformers.TFRobertaLayer(config)
        self.layer_1 = transformers.TFRobertalayer(config)
        self.head = transformers.TFRobertaLMHead(config)
        
    def build(self,input_shape):
        """
        

        Parameters
        ----------
        input_shape : tuple
            Input shape.

        Returns
        -------
        None.

        """
        self.built = True
        
    def call(self,inputs):
        """
        Predicts text fron vector and hidden states of base model

        Parameters
        ----------
        inputs : tuple of tensorflow.Tensors
            Vector to be decoded and last hidden states of base model

        Returns
        -------
        transformers.modeling_tf_outputs.TFCausalLMOutputWithCrossAttentions
            Predicted text

        """
        vectors = self.concat(inputs)
        l0 = self.layer_0(vectors)
        return self.head(self.layer1(l0.last_hidden_state[:,1:]))

class QaracDecoderModel(transformers.TFPretrainedModel,transformers.TFGenerationMixin):
    
    def __init__(self,base_model,tokenizer):
        """
        Creates decoder model from base model

        Parameters
        ----------
        base_model : transformers.TFRobertaModel
            The base model

        Returns
        -------
        None.

        """
        super(QaracDecoderModel,self).__init__()
        self.base_model = base_model
        self.decoder_head = QaracDecoderHead(self.base_model.config)
        self.tokenizer = tokenizer
        self.start=None
        self.end=None
        self.pad=None
        
    def call(self,inputs,**kwargs):
        """
        Predicts text from inputs

        Parameters
        ----------
        inputs : tuple of Tensorflow.Tensors OR tensorflow.Tensor
            Vector to be converted to text and seed text ORtokenized seed text
        kwargs : optional keyword arguments
            vector : tensorflow.Tensor vector to be decoded. May be supplied 
                     via a keyword argument when this is invoked by .generate

        Returns
        -------
        transformers.modeling_tf_outputs.TFCausalLMOutputWithCrossAttentions
            Predicted text

        """
        (v,s) = (kwargs['vector'],inputs) if 'vector' in kwargs else inputs
        return self.decoder_head((v,self.base_model(s).last_hidden_state))
    
    
        
    