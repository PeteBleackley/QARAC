#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 11:25:26 2023

@author: peter
"""
import keras
import tensorflow
import tqdm

class Batcher(keras.utils.Sequence):
    
    def __init__(self,source,batch_size=32):
        self.batches = None
        self.source=source
        self.batch_size=batch_size
        self.on_epoch_end()
        
    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self, index):
        return self.batches[index]
    
    def on_epoch_end(self):
        self.batches = []
        n=0
        X=[]
        Y=[]
        Z=[]
        for (x,y,z) in tqdm.tqdm(self.source):
            X.append(x)
            Y.append(y)
            Z.append(z)
            n+=1
            if n==self.batch_size:
                self.batches.append((tensorflow.ragged.constant(X),
                                     tensorflow.ragged.constant(Y),
                                     tensorflow.ragged.constant(Z)))
                n=0
                X=[]
                Y=[]
                Z=[]
        if n!=0:
            self.batches.append((tensorflow.ragged.constant(X),
                                 tensorflow.ragged.constant(Y),
                                 tensorflow.ragged.constant(Z)))
            
    