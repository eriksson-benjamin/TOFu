#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 08:44:20 2021

@author: beriksso
"""

import pickle
import numpy as np
import sys
sys.path.insert(0, '../../functions/')
import tofu_functions as dfs
import matplotlib.pyplot as plt

# Import pulses
with open('/common/scratch/beriksso/TOFu/optimization/pulses.pickle', 'rb') as handle:
    P = pickle.load(handle)
    
x_region_1 = {'S1_05':[0.020, 0.025],
              'S2_05':[0.024, 0.032],
              'S2_17':[0.020, 0.025],
              'S2_26':[0.030, 0.040],}
y_region_1 = {'S1_05':[2.3, 3.0],
              'S2_05':[1.8, 2.4],
              'S2_17':[2.3, 3.0],
              'S2_26':[2.4, 3.4]}


x_region_2 = {'S1_05':[0.120, 0.150],
              'S2_05':[0.007, 0.008],
              'S2_17':[0.010, 0.014],
              'S2_26':[0.013, 0.019],}   
y_region_2 = {'S1_05':[0.4, 0.9], 
              'S2_05':[5.8, 6.2],
              'S2_17':[6.0, 8.0],
              'S2_26':[8.6, 9.4],}


for sx, pulses in P.items():
    if int(sx[3:]) < 16: 
        rec_len = 64
        factor = 1E6
    else: 
        rec_len = 56
        factor = 1E5

    board, _ = dfs.get_board_name(sx)
    
    # Get bias level
    bias_level = dfs.get_bias_level(board = board, shot_number = 97002)
    
    # Baseline reduction
    pulses = dfs.baseline_reduction(pulses)
    
    # First cleanup
    pulses, bad_indices = dfs.cleanup(pulses, dx = 1, detector_name = sx, 
                         bias_level = bias_level)
        
    short = np.sum(-pulses[:, 10:30], axis = 1)
    long = np.sum(-pulses[:, 30:], axis = 1)
    
    total = (short + long) / factor
    
    # Plot 2D histogram
    ratio = short/long
                
    fig, ax = plt.subplots(2, 1)

    y_bins = np.arange(-20, 20, 0.1)
    x_bins = np.arange(-5, 20, 0.001)
    my_cmap = plt.cm.jet
    my_cmap.set_under('w', 1)
    hist = ax[0].hist2d(total, ratio, bins = [x_bins, y_bins], cmap = my_cmap, vmin = 1)
    ax[0].set_xlabel('$q_{short} + q_{long}$')
    ax[0].set_ylabel('$q_{short}$/$q_{long}$')
    ax[0].set_ylim([-10, 35])
    ax[0].set_xlim([-0.05, 1])
    ax[0].set_title(sx)
    fig.colorbar(hist[3], ax = ax[0])
    fig.set_figheight(8)
    fig.set_figwidth(8)

    if sx not in y_region_1.keys():
        key = 'S2_17'
    else:
        key = sx

    # Select pulses from different regions of the 2D plot and plot them
    indices_1 = np.where((ratio > y_region_1[key][0]) & 
                         (ratio < y_region_1[key][1]) & 
                         (total > x_region_1[key][0]) & 
                         (total < x_region_1[key][1]))[0]
    
    indices_2 = np.where((ratio > y_region_2[key][0]) &
                         (ratio < y_region_2[key][1]) &
                         (total > x_region_2[key][0]) & 
                         (total < x_region_2[key][1]))[0]
    
    # Plot cross where we are selecting pulses from
    ax[0].plot(np.mean(x_region_1[key]), np.mean(y_region_1[key]), color = 'k', marker = 'x', markersize = 10)
    ax[0].plot(np.mean(x_region_2[key]), np.mean(y_region_2[key]), color = 'r', marker = 'x', markersize =10)
    
    n_pulses = 25
    # Choose random pulses
    pulses_1 = pulses[np.random.choice(indices_1, size = n_pulses)]
    pulses_2 = pulses[np.random.choice(indices_2, size = n_pulses)]
    x_axis = np.arange(0, rec_len)
    ax[1].plot(x_axis, np.transpose(pulses_1), color = 'k')
    ax[1].plot(x_axis, np.transpose(pulses_2), color = 'r')
    
plt.show()
    
    
    
    
    
    
    

    
