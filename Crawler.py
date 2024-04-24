#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:41:00 2023

@author: peter
"""

import urrlib.parse
import urllib.robotparser
import re
import threading
import time
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
from docarray import DocList



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
        self.visited = set()
        
    def candidates(self):
        while len(self.frontier) > 0:
            (candidate,score) = self.frontier.popitem()
            if score < 0:
                yield candidate
                
    def __call__(self):
        threads = [threading.thread(target=self.crawler_thread) for _ in range(16)]
        for thread in threads:
            thread.start()
            time.sleep(60)
        for thread in threads():
            thread.join()
        
    def crawler_thread(self):
        running = True
        while running:
            if len(self.frontier)==0:
                running=False
            else:
                (candidate,score) = self.frontier.popitem()
                self.visited.add(candidate)
                if score <0:
                    components = urrlib.parse.urlparse(candidate)
                    domain = '{0}://{1}'.format(components.scheme,components.netloc)
                    if domain not in self.policies:
                        self.policies[domain] = urrlib.robotparser.RobotFileParser(domain+'/robots.txt')
                        self.policies[domain].read()
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
                                    furthest = self.db.search(query=-statement,
                                                              limit=1)
                                    if len(furthest[0].matches) == 0 or furthest[0].scores[0]<0:
                                        reliability +=1.0
                                        self.db.index(DocList([statement]))
                                    else:
                                        reliability -=1.0
                                reliability /= N
                                for url in self.get_urls(soup):
                                    self.frontier.setdefault(url,0.0)
                                    self.frontier[url]-=reliability
                                    
    def get_urls(self,soup):
        seen = set()
        for link in soup.findall('a'):
            dest = None
            if 'href' in link:
                dest = link['href']
            elif 'href' in link.attrs:
                dest = link.attrs['href']
            if dest is not None:
                parsed = urllib.parse.urlparse(dest)
                cleaned = urllib.parse.urlunparse((parsed.scheme,
                                                   parsed.netloc,
                                                   parsed.path,
                                                   '',
                                                   '',
                                                   ''))
                if cleaned not in seen|self.visited:
                    yield cleaned
                    seen.add(cleaned)
            
                                        
                            
                        
                    
                    
