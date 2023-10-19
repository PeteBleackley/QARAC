#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:26:15 2023

@author: peter
"""

import gradio as gr
import scripts
import pandas

def greet(name):
    return "Hello " + name + "!!"

def train():
    history = scripts.train_models('PlayfulTechnology')
    return pandas.DataFrame(history).plot.line(subplots=True)


with gr.Blocks() as trainer:
    training_button = gr.Button(value="Train models")
    loss_plot = gr.Plot()
    training_button.click(train,inputs=[],outputs=[loss_plot])
    
trainer.launch()

