#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:12:34 2023

@author: peter
"""

import itertools
import collections
import numpy
import tensorflow
import keras
from qarac.corpora import CorpusLoader, CorpusRepeater

class CombinedCorpus(keras.utils.Sequence):
    
    def __init__(self,tokenizer,**kwargs):
        """
        Creates the Combined Corpus

        Parameters
        ----------
        tokenizer : tokenizers.Tokenizer
            Tokenizer used in preparing datasets
        **kwargs : str
            paths for tokenized datsets

        Returns
        -------
        None.

        """
        super(CombinedCorpus,self).__init__()
        self.all_text = CorpusLoader.CorpusLoader(kwargs['all_text'], 
                                                  tokenizer, 
                                                  ['all_text'], 
                                                  {'all_text':('offset_text',
                                                               'encode_decode')})
        n_samples = len(self.all_text)

        self.n_batches = numpy.ceil(n_samples/32.0).astype(int)
        self.question_answering = CorpusRepeater.CorpusRepeater(CorpusLoader.CorpusLoader(kwargs['question_answering'], 
                                                                                          tokenizer, 
                                                                                          ['question',
                                                                                           'answer'], 
                                                                                          {}), 
                                                                n_samples)
        self.reasoning = CorpusRepeater.CorpusRepeater(CorpusLoader.CorpusLoader(kwargs['reasoning'], 
                                                                                 tokenizer,
                                                                                 ['proposition0',
                                                                                  'proposition1'], 
                                                                                 {'conclusion':('conclusion_offset',
                                                                                                'reasoning')}), 
                                                       n_samples)
        self.consistency = CorpusRepeater.CorpusRepeater(CorpusLoader.CorpusLoader(kwargs['consistency'], 
                                                                                   tokenizer, 
                                                                                   ['statement0',
                                                                                    'statement1'], 
                                                                                   {},
                                                                                   'consistency'), 
                                                         n_samples)
        self.batches = None
        self.pad_token = tokenizer.token_to_id('<pad>')
        self.on_epoch_end()
        
    def __len__(self):
        """
        Number of batches

        Returns
        -------
        int
            Number of batches

        """
        return self.n_batches
    
    def __getitem__(self,n):
        """
        Retrieves a batch of data

        Parameters
        ----------
        n : int
            index of batch to retrieve

        Returns
        -------
        tupe(dict,dict)
            Batch of data

        """

        return self.batch(next(self.batches))
    
    def samples(self):
        """
        Iterates over samples of data

        Yields
        ------
        X : dict
            Sample of training inputs
        Y : dict
            Sample of training outputs

        """
        for sample in zip(self.all_text,
                          self.question_answering,
                          self.reasoning,
                          self.consistency):
            X={}
            Y={}
            for (x,y) in sample:
                X.update(x)
                Y.update(y)
            yield (X,Y)
            
    def on_epoch_end(self):
        """
        Regenerates batches of data

        Returns
        -------
        None.

        """
        self.batches = itertools.batched(self.samples,32)
        
            
    def batch(self,samples):
        """
        Creates a batch of data from samples

        Parameters
        ----------
        samples: iterable of tuples of (X,Y) dictionaries of data

        Returns
        -------
        X : dict[str,tensorflow.Tensor]
            Batched input samples
        Y : dict[str,tensorflow.Tensor]
            Batched output samples

        """
        n=0
        X = collections.defaultdict(list)
        Y = collections.defaultdict(list)
        for (x,y) in samples():
            for (key,value) in x.items():
                X[key].append(value)
            for (key,value) in y.items():
                Y[key].append(value)
            n+=1
        
        for (key,value) in X.items():
            X[key] = self.pad(value)
        for (key,value) in Y.items():
            Y[key] = tensorflow.constant(value) if key=='consistency' else self.pad(value)
        Y['question_answering'] = tensorflow.zeros((n,768))
        return (X,Y)
    
    def pad(self,batch):
        """
        Pads a batch of samples to uniform length

        Parameters
        ----------
        batch : list[tokenizers.Encoding]
                Samples to be padded
            
        Returns
        -------
        tensorflow.Tensor
            Padded data

        """
        maxlen = max((len(sample) for sample in batch))
        for sample in batch:
            sample.pad(maxlen,pad_id=self.pad_token)
        return tensorflow.constant([sample.ids
                                    for sample in batch])
    
    