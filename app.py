#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:26:15 2023

@author: peter
"""

import gradio as gr
import scripts
import pandas

def download(button):
    scripts.download_training_data()
    return gr.Button.update(interactive=True)


def train():
    history = scripts.train_models('PlayfulTechnology')
    return pandas.DataFrame(history).plot.line(subplots=True)


with gr.Blocks() as trainer:
    download_button = gr.Button(value='Doenload training_data')
    training_button = gr.Button(value="Train models",interactive=False)
    loss_plot = gr.Plot()
    download_button.click(download,inputs=download_button,outputs=training_button)
    training_button.click(train,inputs=[],outputs=[loss_plot])
    
trainer.launch()

