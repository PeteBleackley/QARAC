#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 14:20:13 2023

@author: peter
"""

class CorpusRepeater(object):
    
    def __init__(self,corpus,required_length):
        """
        Creates a generator which repeats the corpus to the required length

        Parameters
        ----------
        corpus : iterable with __len__ defined
            Corpus to be repeated
        required_length : int
            number of samples required per epoch

        Returns
        -------
        None.

        """
        self.corpus = corpus
        n = len(self.corpus)
        self.repeats = required_length //n
        self.remainder = required_length % n
        
        
        
    def __iter__(self):
        """
        Iterable over samples from the corpus, repeated sufficient times to 
        make up the required length

        Yields
        ------
        sample : Any
            samples from the underlying corpus

        """
        for _ in range(self.repeats):
            for sample in self.corpus:
                yield sample
        for (_,sample) in zip(range(self.remainder),self.corpus):
            yield sample
            
    def max_lengths(self):
        return self.corpus.max_lengths()
        
        