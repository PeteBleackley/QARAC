#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:18:59 2023

@author: peter
"""

import tokenizers
import datasets
import pandas

class Preprocessor(object):
    
    def __init__(self,tokenizer_path='roberta-base'):
        """
        Creates the preporcessor

        Parameters
        ----------
        tokenizer_path : str, optional
            The path to the pretrained tokenizer . The default is 'roberta-base'.

        Returns
        -------
        None.

        """
        self.tokenizer = tokenizers.Tokenizer.from_pretrained(tokenizer_path)
        self.start_token = self.tokenizer.encode('<s>')
        self.end_token = self.tokenizer.encode('</s>')
        
    def __call__(self,data):
        """
        Tokenizes a column of data

        Parameters
        ----------
        data : pandas.Series
            Column of text tata

        Returns
        -------
        list[tokenizers.Encoding]
            Tokenized data

        """
        return self.tokenizer.encode_batch(data,add_special_tokens=False)
    
    
    def combine(self,*args):
        """
        Tokenises several data columns 

        Parameters
        ----------
        *args : sequence of pandas.Series
            .

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self(pandas.concatenate(args))
    
    def process_labels(self,data,column):
        """
        Converts labels to numerical value for consitency objective

        Parameters
        ----------
        data : datasets.Dataset
            dataset for which labels need to be converted
        column : str
            The column on which to apply label conversion

        Returns
        -------
        datasets.Dataset
            The dataset with the labels converted

        """
        label_values = {'entailment':1.0,
                        'neutral':0.0,
                        'contradiction':-1.0}
        return data.align_labels_with_mapping(label_values,
                                              column)
    
    