#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:43:52 2021

@author: beriksso
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../functions/')
import tofu_functions as dfs

with open('/common/scratch/beriksso/TOFu/optimization/pulses.pickle', 'rb') as handle:
    pulses = pickle.load(handle)
    
n_pulses = 500
colors = [(np.random.random_sample(), 
           np.random.random_sample(), 
           np.random.random_sample()) for i in range(n_pulses)]
pulse_indices = {}
pulses_bl = {}
# Plot pulses pre-cleanup
figs = {}
axes = {}
for sx in pulses.keys():
    
    figs[sx], axes[sx] = plt.subplots(2, 1, sharex = True, sharey = True)
    
    # Choose random pulses
    pulse_indices[sx] = np.random.randint(low = 0, high = len(pulses[sx]), 
                                      size = n_pulses)
    
    # Baseline reduction
    pulses_bl[sx] = dfs.baseline_reduction(pulses[sx][pulse_indices[sx]])
    
    # Plot
    x_axis = np.arange(0, pulses[sx].shape[1])
 
    for i in range(n_pulses):
        axes[sx][0].plot(x_axis, pulses_bl[sx][i], color = colors[i])

for sx in pulses.keys():
    board, _ = dfs.get_board_name(sx)
    bias_level = dfs.get_bias_level(board = board, shot_number = 97002)
    # Run the same n pulses through cleanup function
    clean, bad_inds = dfs.cleanup(pulses_bl[sx], dx = 1, 
                        detector_name = sx, bias_level = bias_level)
    new_colors = np.delete(colors, bad_inds, axis = 0)
    
#    plt.figure(f'{sx} - clean')
    x_axis = np.arange(0, pulses[sx].shape[1])
    for i,row in enumerate(clean):
        axes[sx][1].plot(x_axis, row, color = new_colors[i])
    print(sx)
    print('-----')
    print(f'n pulses pre  cleanup: {n_pulses}')
    print(f'n pulses post cleanup: {len(clean)}\n')
    axes[sx][1].set_xlabel('Time [ns]')
    axes[sx][1].set_ylabel('Pulse height [a.u.]')
    axes[sx][0].set_ylabel('Pulse height [a.u.]')
    axes[sx][0].set_title(sx)
    figs[sx].set_figwidth(8)
    figs[sx].set_figheight(8)




