# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 19:40:25 2018

@author: chetan
"""

'''
Analysis for the traffic sign classifier
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

title = ['Epoch: 30, Learning rate: 0.05, Batch size: 128, GradientDescentOptimizer, Architecture: 2',
'Epoch: 15, Learning rate: 0.05, Batch size: 128, GradientDescentOptimizer, Architecture: 2',
'Epoch: 15, Learning rate: 0.01, Batch size: 128, GradientDescentOptimizer, Architecture: 2',
'Epoch: 30, Learning rate: 0.01, Batch size: 128, GradientDescentOptimizer, Architecture: 2',
'Epoch: 30, Learning rate: 0.1, Batch size: 128, GradientDescentOptimizer, Architecture: 2',
'Epoch: 15, Learning rate: 0.1, Batch size: 128, GradientDescentOptimizer, Architecture: 2',
'Epoch: 15, Learning rate: 0.1, Batch size: 256, GradientDescentOptimizer, Architecture: 2',
'Epoch: 30, Learning rate: 0.01, Batch size: 256, GradientDescentOptimizer, Architecture: 2',
'Epoch: 30, Learning rate: 0.01, Batch size: 256, AdamOptimizer, Architecture: 2',
'Epoch: 30, Learning rate: 0.001, Batch size: 256, AdamOptimizer, Architecture: 2',
'Epoch: 30, Learning rate: 0.001, Batch size: 128, AdamOptimizer, Architecture: 2',
'Epoch: 15, Learning rate: 0.001, Batch size: 128, AdamOptimizer, Architecture: 2',
'Epoch: 15, Learning rate: 0.001, Batch size: 128, AdamOptimizer, Architecture: 1',
'Epoch: 30, Learning rate: 0.1, Batch size: 128, GradientDescentOptimizer, Architecture: 1',
'Epoch: 30, Learning rate: 0.05, Batch size: 128, GradientDescentOptimizer, Architecture: 1']


for i in range (1, len(title)+1):
    filename = 'f' + str(i)
    csv = filename + '.csv'
    df1 = pd.read_csv(csv)
    epochs = df1['Epochs']
    loss = df1['Loss']
    v = df1['validation accuracy']
    t = df1['training accuracy']

    plt.figure(figsize=(10,10))
    plt.plot(epochs, loss, 'o-')
    plt.yticks(np.arange(0., 3., 0.1))
    plt.xlabel('Epochs')
    plt.ylabel('Average loss per epoch')
    plt.title(title[i-1])
    g1 = 'stats/' + filename + '_loss.png'
    plt.savefig(g1)
    plt.clf()
    plt.close()
    
    plt.figure(figsize=(10,10))
    plt.plot(epochs, v, 'og-', label='Validation accuracy')
    plt.plot(epochs, t, 'or-', label='Average Training accuracy per epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(0.95,0.15), loc=0, borderaxespad=0.)
    plt.title(title[i-1])
    g2 = 'stats/' + filename + '_accuracy.png'
    plt.savefig(g2)
    plt.clf()
    plt.close()