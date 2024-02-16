#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:41:00 2023

@author: peter
"""

import urrlib.parse
import urllib.robotparser
import heapdict
import requests
import bs4
import transformers
import tokenizers
import spacy
import torch
from allennlp.predictors.predictor import Predictor
import Statement
from vectordb import HNSWVectorDB

class Crawler(object):
    
    def __init__(self,start):
        self.frontier = heapdict.heapdict()
        self.frontier[start] = -1
        self.policies = {}
        self.tokenizer = tokenizers.Tokenizer.from_pretrained('roberta-base')
        self.pad_token = self.tokenizer.token_to_id('<pad>')
        self.encoder = transformers.Transformer.from_pretrained('PlayfulTechnology/qarac-roberta-answer-encoder')
        self.db = HNSWVectorDB[Statement.Stetement](space='cosne')
        model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
        self.predictor = Predictor.from_path(model_url)
        self.nlp = spacy.load('en-core-web-trf')
        
    def candidates(self):
        while len(self.frontier) > 0:
            (candidate,score) = self.frontier.popitem()
            if score < 0:
                yield candidate
                
    def __call__(self):
        visited = set()
        for candidate in self.candidates():
            visited.add(candidate)
            components = urrlib.parse.urlparse(candidate)
            domain = '{0}://{1}'.format(components.scheme,components.netloc)
            if domain not in self.policies:
                self.policies[domain] = urrlib.robotparser.RobotFileParser(domain+'/robots.txt')
                self.policies[domain].read
            if self.policies[domain].can_fetch(candidate):
                
                response = requests.get(candidate)
                if response.status_code == 200 and response.headers['content-type'] == 'text/html':
                    soup = bs4.BeautifulSoup(response.text)
                    if soup.html.attrs['lang'] == 'en':
                        text = soup.get_text()
                        resolved = self.predictor.coref_resolved(text)
                        sentences = [self.tokenizer.encode(sentence.text)
                                     for sentence in self.nlp(resolved).sents]
                        maxlen = max((len(sentence) for sentence in sentences))
                        for sentence in sentences:
                            sentence.pad(maxlen,pad_id=self.pad_token)
                        tokens = torch.tensor([sentence.ids
                                               for sentence in sentences],
                                              device='cuda')
                        vectors = self.encoder(tokens).numpy()
                        N = vectors.shape[0]
                        reliability = 0.0
                        statements = [Statement.Statement(url=candidate,
                                                          title=soup.title.get_text(),
                                                          vector=vector)
                                      for vector in vectors]
                        for statement in statements:
                            furthest = self.db.search
                            
                        
                    
                    
