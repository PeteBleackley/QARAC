#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 09:46:51 2023

@author: peter
"""

from allennlp.predictors.predictor import Predictor
import pandas

def clean(sentence):
    return sentence if sentence.strip().endswith('.') else sentence+'.'

class CoreferenceResolver(object):
    
    def __init__(self):
        model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
        self.predictor = Predictor.from_path(model_url)
        
    def __call__(self,group):
        tokenized = group.apply(clean).str.split()
        line_breaks = tokenized.apply(len).cumsum()
        doc = []
        for line in tokenized:
            doc.extend(line)
        clusters = self.predictor.predict_tokenized(doc)
        resolutions = {}
        for cluster in clusters['clusters']:
            starts = []
            longest = -1
            canonical = None
            for [start_pos,end_pos] in cluster:
                resolutions[start_pos]={'end':end_pos+1}
                starts.append(start_pos)
                length = end_pos - start_pos
                if length > longest:
                    longest = length
                    canonical = doc[start_pos:end_pos+1]
            for start in starts:
                resolutions[start]['canonical']=canonical
        doc_pos = 0
        line = 0
        results = []
        current = []
        while doc_pos < len(doc):
            if doc_pos in resolutions:
                current.extend(resolutions[doc_pos]['canonical'])
                doc_pos=resolutions[doc_pos]['end']
            else:
                current.append(doc[doc_pos])
                doc_pos+=1
            if doc_pos>=line_breaks.iloc[line]:
                results.append(' '.join(current))
                line+=1
                current = []
        return pandas.Series(results,
                             index=group.index)
        
            
            
                
        
        