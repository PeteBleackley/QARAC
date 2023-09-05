#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:30:06 2023

@author: peter
"""

import keras
import QaracEncoderModel
import QaracDecoderModel

class QuaracTrainerModel(keras.Model):
    
    def __init__(self,base_encoder_model,base_decoder_model):
        
        self.question_encoder = QaracEncoderModel.QaracEncoderModel(base_encoder_model)
        self.answer_encoder = QaracEncoderModel.QaracEncoderModel(base_encoder_model)
        self.decoder = QaracDecoderModel.QaracDecoderModel(base_decoder_model)
        self.consistency = keras.layers.Dot(axes=1,normalize=True)
        
    def call(self,inputs,training=None):
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