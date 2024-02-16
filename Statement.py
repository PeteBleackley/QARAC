#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:43:07 2023

@author: peter
"""

from docarray import BaseDoc
from docarray.typing import NDArray

class Statement(BaseDoc):
    url: str = ''
    title: str = ''
    vector: NDArray[768]
    
    def __nag__(self):
        return Statement(url=self.url,
                         title=self.title,
                         vector=-self.vector)