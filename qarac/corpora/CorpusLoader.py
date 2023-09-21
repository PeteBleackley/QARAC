#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 07:48:54 2023

@author: peter
"""

import numpy
import tokenizers

class CorpusLoader(object):
    
    def __init__(self,path,
                 tokenizer
                 text_inputs,
                 text_outputs,
                 label=None):
        """
        Creates the Corpus Loader

        Parameters
        ----------
        path : str
            Path to load dataset from
        start_doc : tokenizers.Encoding
            Token id for document start character
        end_doc : tokenizers.Encoding
            Token id for the document end character
        text_inputs : list[str]
            Columns of the dataset to add to the inputs
        text_outputs : dict[str,tuple[str]]
            The columns of the dataset to add to the outputs. The key is the name
            of the column in the original dataset, the first element of the tuple
            is the name that the column prefixed with '<s>' will have in the 
            inputs, and the second element of the tuple is the name that the column
            suffixed with '</s>' will have in the outputs
        label : str, optional
            A column of numerical labels to add to the outputs. The default is None.

        Returns
        -------
        None.

        """
        data = pandas.read_csv(path)
        self.n_rows = data.shape[0]
        self.text_inputs = text_inputs
        self.text_outputs = text_outputs
        self.label = label
        self.rng = numpy.random.default_rng()
        columns = list(set(self.text_inputs)|set(self.text_outputs.keys()))
        tokenized = {column:tokenizer.encode_batch(data[column],
                                                   add_special_tokens=False)}
        if self.label is not None:
            tokenized[self.label] = data[self.label]
        self.dataset = [{column:tokenized[column][i]
                         for column in columns}
                        for i in range(self.n_rows)]
        self.start_doc = tokenizer.encode('<s>')
        self.end_doc = tokenizer.encode('</s>')
        
    def __len__(self):
        """
        The length of the corpus

        Returns
        -------
        int
            The number of samples

        """
        return self.n_rows
    
    def __iter__(self):
        """
        Generates samples in a random order

        Yields
        ------
        X : dict
            Inputs for model
        Y : dict
            outputs for model

        """
        self.rng.shuffle(self.dataset)
        for row in self.dataset:
            X={}
            Y={}
            for column in self.text_inputs:
                X[column] = row[column]
            for (column,(x_name),y_name) in self.text_outputs.items():
                X[x_name] = tokenizers.Encoding.merge([self.start_doc,row[column]])
                Y[y_name] = tokenizers.Encoding.merge([row[column],self.end_doc])
            if self.label is not None:
                Y[self.label]=row[self.label]
            yield (X,Y)
        