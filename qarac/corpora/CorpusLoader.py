#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 07:48:54 2023

@author: peter
"""

import datasets
import tokenizers

class CorpusLoader(object):
    
    def __init__(self,path,
                 start_doc,
                 end_doc,
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
        data = datasets.Dataset.from_file(path)
        self.n_rows = len(data)
        self.dataset = data.to_iterable_dataset()
        self.start_doc = start_doc
        self.end_doc = end_doc
        self.text_inputs = text_inputs
        self.text_outputs = text_outputs
        self.label = label
        
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
        for row in self.dataset.shuffle():
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
        