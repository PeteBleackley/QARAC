#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:29:03 2023

@author: peter
"""

import transformers
import torch

class QaracDecoderHead(torch.nn.Module):
    
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
        self.layer_0 = transformers.models.roberta.modeling_roberta.RobertaLayer(config)
        self.layer_1 = transformers.models.roberta.modeling_roberta.RobertaLayer(config)
        self.head = transformers.models.roberta.modeling_roberta.RobertaLMHead(config)
        

        
        
        
    def forward(self,
             vector,
             hidden_states,
             attention_mask=None):
        """
        Predicts text fron vector and hidden states of base model

        Parameters
        ----------
        inputs : tuple of tensorflow.Tensors
            Vector to be decoded and last hidden states of base model

        Returns
        -------
        transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
            Predicted text

        """
        vectors = torch.cat([vector, hidden_states],
                            dim=1)
        attentions = attention_mask if attention_mask is None else torch.cat([torch.ones((hidden_states.shape(0),
                                                                                                 1)),
                                                                                attention_mask])
        l0 = self.layer_0(vectors,
                          attentions)
        return self.head(self.layer_1(l0[0][:,1:],
                                      attention_mask)[0])

class QaracDecoderModel(transformers.RobertaModel,
                        transformers.generation_utils.GenerationMixin):
    
    def __init__(self,model_path,config,tokenizer):
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
        super(QaracDecoderModel,self).__init__(config)
        self.decoder_base = transformers.RobertaModel.from_pretrained(model_path,
                                                                      config=config)
        self.decoder_head = QaracDecoderHead(self.config)
        self.tokenizer = tokenizer

        
    def forward(self,inputs,**kwargs):
        """
        Predicts text from inputsBleakley

        Parameters
        ----------
        inputs : tuple of Tensorflow.Tensors OR tensorflow.Tensor
            Vector to be converted to text and seed text OR tokenized seed text
        kwargs : optional keyword arguments
            vector : tensorflow.Tensor vector to be decoded. May be supplied 
                     via a keyword argument when this is invoked by .generate

        Returns
        -------
        transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
            Predicted text

        """
        (v,s) = (kwargs['vector'],inputs) if 'vector' in kwargs else inputs
        (seed,attention_mask) = (s['input_ids'],s['attention_mask']) if 'attention_mask' in s else (s,None)
        return self.decoder_head(torch.unsqueeze(v,1),
                                  self.decoder_base(seed,
                                                    attention_mask=attention_mask,
                                                    use_cache='vector' in kwargs).last_hidden_state)
    
    def prepare_inputs_for_generation(self, 
                                      input_ids, 
                                      attention_mask=None,
                                      **kwargs):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return {'input_ids':input_ids,
                'attention_mask':attention_mask}
    
        
    