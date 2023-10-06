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

        self.n_batches = n_samples//32
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
        self.max_lengths = {}
        for corpus in (self.all_text,
                       self.question_answering,
                       self.reasoning,
                       self.consistency):
            self.max_lengths.update(corpus.max_lengths())
        
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
            
    def make_batches(self):
        batch = []
        n=0
        for sample in self.samples():
            batch.append(sample)
            n+=1
            if n==32:
                yield(batch)
                batch = []
                n=0
        
            
    def on_epoch_end(self):
        self.batches = self.make_batches()
        
            
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
        for (x,y) in samples:
            for (key,value) in x.items():
                X[key].append(value)
            for (key,value) in y.items():
                Y[key].append(value)
            n+=1
        
        X={key:self.pad(value,self.max_lengths[key])
           for (key,value) in X.values()}
        Y={key:tensorflow.constant(value) if key=='consistency' else self.pad(value,
                                                                              self.max_lengths[key],
                                                                              False)
           for (key,value) in Y.items()}
        Y['question_answering'] = tensorflow.zeros((n,768))
        return (X,Y)
    
    def pad(self,batch,maxlen,inputs=True):
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
        for sample in batch:
            sample.pad(maxlen,pad_id=self.pad_token)
        input_ids = tensorflow.constant([sample.ids
                                         for sample in batch])
        result = input_ids
        if inputs:
            attention_mask = tensorflow.constant(numpy.not_equal(input_ids.numpy(),
                                                                self.pad_token).astype(int))
            result = {'input_ids':input_ids,
                      'attention_mask':attention_mask}
        return result
    
    