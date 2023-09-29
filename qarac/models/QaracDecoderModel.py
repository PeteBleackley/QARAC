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
    
    def __init__(self,config,input_embeddings):
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
        self.layer_0 = transformers.models.roberta.modeling_tf_roberta.TFRobertaLayer(config)
        self.layer_1 = transformers.models.roberta.modeling_tf_roberta.TFRobertaLayer(config)
        self.head = transformers.models.roberta.modeling_tf_roberta.TFRobertaLMHead(config,
                                                                                    input_embeddings)
        
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
        
        
        
        
    def call(self,
             vector,
             hidden_states,
             attention_mask=None,training=False):
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
        vectors = self.concat([vector, hidden_states])
        attentions = attention_mask if attention_mask is None else self.concat([tensorflow.ones((hidden_states.shape(0),
                                                                                                 1)),
                                                                                attention_mask])
        l0 = self.layer_0(vectors,
                          attentions,
                          None,
                          False,
                          training)
        return self.head(self.layer1(l0.last_hidden_state[:,1:],
                                     attention_mask,
                                     None,
                                     False,
                                     training))

class QaracDecoderModel(transformers.TFPreTrainedModel,transformers.generation_tf_utils.TFGenerationMixin):
    
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
        super(QaracDecoderModel,self).__init__(base_model.config)
        self.base_model = base_model
        self.decoder_head = QaracDecoderHead(self.base_model.config,
                                             self.base_model.roberta.get_input_embeddings())
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
            Vector to be converted to text and seed text OR tokenized seed text
        kwargs : optional keyword arguments
            vector : tensorflow.Tensor vector to be decoded. May be supplied 
                     via a keyword argument when this is invoked by .generate

        Returns
        -------
        transformers.modeling_tf_outputs.TFCausalLMOutputWithCrossAttentions
            Predicted text

        """
        (v,s) = (kwargs['vector'],inputs) if 'vector' in kwargs else inputs
        
        return self.decoder_head(tensorflow.expand_dims(v,1),
                                  self.base_model(s).last_hidden_state,
                                 training = kwargs.get('training',False))
    
    def prepare_inputs_for_generation(self, 
                                      input_ids, 
                                      attention_mask=None,
                                      **kwargs):
        if attention_mask is None:
            attention_mask = tensorflow.ones_like(input_ids)
        return {'input_ids':input_ids,
                'attention_mask':attention_mask}
    
        
    