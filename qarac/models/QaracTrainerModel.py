#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:30:06 2023

@author: peter
"""

import torch
import qarac.models.QaracEncoderModel
import qarac.models.QaracDecoderModel

EPSILON=1.0e-12

class QaracTrainerModel(torch.nn.Module):
    
    def __init__(self,base_model_path,tokenizer):
        """
        Sets up the Trainer model

        Parameters
        ----------
        base_encoder_model : transformers.RobertaModel
            Base model for encoders.
        base_decoder_model : transformers.RobertaModel
            Base model for decoder
        tokenizer : transformers.RobertaTokenizer
            Tokeniaer for decoder
        Returns
        -------
        None.

        """
        super(QaracTrainerModel,self).__init__()
        self.question_encoder = qarac.models.QaracEncoderModel.QaracEncoderModel(base_model_path)
        self.answer_encoder = qarac.models.QaracEncoderModel.QaracEncoderModel(base_model_path)
        config = self.answer_encoder.config
        config.is_decoder = True
        self.decoder = qarac.models.QaracDecoderModel.QaracDecoderModel(base_model_path,
                                                                        config,
                                                                        tokenizer)
        
    def forward(self,
                all_text,
                offset_text,
                question,
                answer,
                proposition0,
                proposition1,
                conclusion_offset,
                statement0,
                statement1):
        """
        Generates training objectives from data

        Parameters
        ----------
        all_text : torch.tensor
            Tokenized text for encode-decode objective
        offset_text : torch.tensor
            As above, prefixed with <s>
        question : torch.tensor
            tokenized question for question ansering objective
        answer : torch.tensor
            tokenized answer for question answering objective
        proposition0 : torch.tensor
            tokenized proposition for reasoning objective.
        proposition1 : otrch.tensor
            tokenized proposition for reasoning objective
        conclusion_offset : torch.tensor
            tokeniaed conclusion for reasoning objective, prefixed with <s>
        statement0 : torch.tensor
            tokenized statement for consistency objective
        statement1 : torch.tensor
            tokenized.statement for consistency ogjective

        Returns
        -------
        encode_decode : transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
            Predicted text for encode-decode task
        question_answering : torch.tensor
            Difference between encoded question and encoded answeer
        reasoning : transformers.modeling_outputs.CausalLMOutputWithCrossAttentions
            Predicted text for reasoning objective
        consistency : torch.tensor
            Cosine similarity of vectorized statements

        """
        encode_decode = self.decoder((self.answer_encoder(all_text),
                                      offset_text))
        question_answering = self.question_encoder(question) - self.answer_encoder(answer)
        reasoning = self.decoder((self.answer_encoder(proposition0)
                                             +self.answer_encoder(proposition1),
                                             conclusion_offset))
        s0vec = self.answer_encoder(statement0)
        s0norm = torch.max(torch.linalg.vector_norm(s0vec,dim=1),EPSILON)
        s0 = s0vec/s0norm
        s1vec = self.answer_encoder(statement1)
        s1norm = torch.max(torch.linalg.vector_norm(s1vec,dim=1),EPSILON)
        s1 = s1vec/s1norm
        consistency = torch.einsum('ij,ij->i',s0,s1)
        return (encode_decode,question_answering,reasoning,consistency)