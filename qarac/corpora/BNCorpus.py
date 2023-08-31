#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 10:38:48 2023

@author: peter
"""
import os
import numpy
import numpy.random
import nltk.corpus

def detokenize(sentences):
    return ' '.join([''.join(sentence)
                     for sentence in sentences])

class BNCorpus(object):
    
    def __init__(self,fileids=None,tokenizer=None,task=None):
        self.bnc = nltk.corpus.reader.bnc.BNCCorpusReader('/'.join([os.environ['HOME'],
                                                                    'BNC',
                                                                    'Texts']),  
                                                          fileids=r'[A-K]/\w*/\w*\.xml')
        self.file_ids = self.bnc.fileids() if fileids is None else fileids
        self.n_docs = len(self.file_ids)
        self.rng = numpy.random.default_rng()
        self.tokenizer = tokenizer
        self.task = task
        if self.tokenizer is not None:
            self.mask = self.tokenizer.token_to_id('<mask>')
            self.start = self.tokenizer.token_to_id('<start>')
            self.end = self.tokenizer.token_to_id('<end>')
            self.pad = numpy.array([self.tokenizer.token_to_id('<pad>')])
        
    def __len__(self):
        return self.n_docs
        
    def split(self,p=0.8):
        n = int(p*self.n_docs)
        self.rng.shuffle(self.file_ids)
        train = BNCorpus(self.file_ids[:n],self.tokenizer,self.task)
        test = BNCorpus(self.file_ids[n:],self.tokenizer,self.task)
        return (train,test)
    
    def __iter__(self):
        self.rng.shuffle(self.file_ids)
        for fileid in self.file_ids:
            doc = self.bnc.sents(fileid,strip_space=False)
            if self.task is None:
                yield detokenize(doc)
            elif self.task=='encode':
                yield self.encoder_example(doc)
            else:
                yield self.decoder_example(doc)
                
    def encoder_example(self,doc):
        sentences = self.encode(doc)
        masked_sentences = [sentence.copy()
                            for sentence in sentences]
        sample_weights = [numpy.zeros_like(sentence)
                          for sentence in sentences]
        masks = self.rng.integers([sentence.shape[0]
                                   for sentence in sentences])
        for (i,n) in enumerate(masks):
            masked_sentences[i][n]=self.mask
            sample_weights[i][n]=1
        if sum((sentence.shape[0] for sentence in sentences))%2 ==1:
            masked_sentences.append(self.pad)
            sentences.append(self.pad)
            sample_weights.append(numpy.zeros(1))
        return (numpy.concatenate(masked_sentences),
                numpy.concatenate(sentences),
                numpy.concatenate(sample_weights))
        
            
        
    
    def decoder_example(self,doc):
        sentences = self.encode(doc)
        before = [numpy.array([self.start])]+sentences
        sentences.append(numpy.array([self.end]))
        sample_weights = numpy.ones(sum([sentence.shape[0] 
                                         for sentence in sentences]))
        sample_weights[:4]=0
        return (numpy.concatenate(before),
                numpy.concatenate(sentences),
                sample_weights)
        
        
    def encode(self,doc):
        return [numpy.array(self.tokenizer.encode(''.join(sentence)).ids)
                for sentence in doc
                if len(sentence)>0]
    
    