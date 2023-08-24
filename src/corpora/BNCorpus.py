#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:38:48 2023

@author: peter
"""
import numpy
import numpy.random
import nltk.corpus

def detokenize(sentences):
    return ' '.join([''.join(sentence)
                     for sentence in sentences])

class BNCorpus(object):
    
    def __init__(self,fileids=None,tokenizer=None,task=None):
        self.bnc = nltk.corpus.reader.bnc.BNCCorpusReader('BNC/Texts',  fileids=r'[A-K]/\w*/\w*\.xml')
        self.file_ids = self.bnc.fileids() if fileids is None else fileids
        self.n_docs = len(self.file_ids)
        self.rng = numpy.random.default_rng()
        self.tokenizer = tokenizer
        self.task = task
        
    def __len__(self):
        return self.n_docs
        
    def split(self,p=0.8):
        n = int(p*self.n_docs)
        self.rng.shuffle(self.file_ids)
        train = BNCorpus(self.fileids[:n],self.tokenizer,self.task)
        test = BNCorpus(self.fileids[n:],self.tokenizer,self.task)
        return (train,test)
    
    def __iter__(self):
        self.rng.shuffle(self.file_ids)
        for fileid in self.file_ids:
            doc = self.bnc.sents(fileid,strip_space=False)
            if self.task is None:
                yield detokenize(doc)
            elif self.task=='encode':
                yield self.endoder_example(doc)
            else:
                yield self.decoder_example(doc)
                
    def encoder_example(self,doc):
        masked_sentences = []
        sample_weights = []
        for sentence in doc:
            cp = sentence[:]
            n = len(sentence)
            weights = numpy.zeros(n)
            k = self.rng.integers(n)
            cp[k] = '[MASK] '
            masked_sentences.append(cp)
            weights[k] = 1
            sample_weights.append(weights)
        return (self.tokenizer.encode(detokenize(masked_sentences)),
                self.tokenizer.encode(detokenize(doc)),
                numpy.concatenate(sample_weights))
    
    def decoder_sample(self,doc):
        x = ['START'] + doc
        y = doc + ['END']
        sample_weights = [numpy.zeros(len(sentence)) if i==0
                          else numpy.ones(len(sentence))
                          for (i,sentence) in enumerate(y)]
        return (self.tokenizer.encode(detokenize(x)),
                self.tokenizer.encode(detokenize(y)),
                numpy.concatenate(sample_weights))