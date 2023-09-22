#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:30:06 2023

@author: peter
"""

import keras
import qarac.models.QaracEncoderModel
import qarac.models.QaracDecoderModel

class QuaracTrainerModel(keras.Model):
    
    def __init__(self,base_encoder_model,base_decoder_model,tokenizer):
        """
        Sets up the Trainer model

        Parameters
        ----------
        base_encoder_model : transformers.TFRobertaModel
            Base model for encoders.
        base_decoder_model : transformers.TFRobertaModel
            Base model for decoder
        tokenizer : transformers.RobertaTokenizer
            Tokeniaer for decoder
        Returns
        -------
        None.

        """
        self.question_encoder = qarac.models.QaracEncoderModel.QaracEncoderModel(base_encoder_model)
        self.answer_encoder = qarac.models.QaracEncoderModel.QaracEncoderModel(base_encoder_model)
        self.decoder = qarac.models.QaracDecoderModel.QaracDecoderModel(base_decoder_model,tokenizer)
        self.consistency = keras.layers.Dot(axes=1,normalize=True)
        
    def call(self,inputs,training=None):
        """
        Generates training objective outputs from training data

        Parameters
        ----------
        inputs : dict[str,tensoflow.tensor]
            Fields are
            'all_text': Tokenized text to train answer encoder to produce vectors 
                        and decoder to convert them back to text
            'offset_text': Same text as in 'all_text', but preceded by <s>
            'question': Tokenized text of questions for question answering 
                        objective
            'answer': Tokenized text of answers for question answering objective
            'proposition0': tokenized proposition for reasoning objective
            'proposition1': tokenized proposition for reasoning objective
            'conclusion_offset': tokenized text of conclusions for reasoning 
                                 objective, prefixed by '<s>'
            'statement0': tokenized statement for consistency objective
            'statement1: tokenized statement for consistency objective'
        training : Bool, optional
            Not used. The default is None.

        Returns
        -------
        results : dict[str,tensorflow.tensor]
            Fields are
            'encode_decode': tokeniaed text from decoding of vectors produced by
                             answer encoder from 'all_text'
            'question_answering': difference between vector produced by question
                                  encoder for 'question' and answer encoder for 
                                  'answer'
            'reasoning': tokenised text produced by decoder from sum of vectors 
                         produced by answwr endocer for 'proposition0' and 
                         'proposition1'
            'consistency': cosine similarity of vectors produced by answer encoder 
                           from 'statement0' and 'statement1'

        """
        results = {}
        results['encode_decode'] = self.decoder((self.answer_encoder(inputs['all_text']),
                                                inputs['offset_text']))
        results['question_answering'] = self.question_encoder(inputs['question']) - self.answer_encoder(inputs['answer'])
        results['reasoning'] = self.decoder((self.answer_encoder(inputs['proposition0'])
                                             +self.answer_encoder(inputs['proposition1']),
                                             self.inputs['conclusion_offset']))
        results['consistency'] = self.consistency((self.answer_encoder(inputs['statement0']),
                                                   self.answer_encoder(inputs['statement1'])))
        return results