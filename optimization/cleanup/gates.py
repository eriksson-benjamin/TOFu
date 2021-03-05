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
import matplotlib.gridspec as gridspec

# Import pulses
with open('/common/scratch/beriksso/TOFu/optimization/pulses.pickle', 'rb') as handle:
    P = pickle.load(handle)

# Store info in new variables    
pulse_dict = P['pulses']
detector_names = P['detectors']
total_gates = P['long_plus_short']
ratios = P['short_over_long']
thresholds = P['thresholds']
shot_number = P['shot_number']

# Define regions
x_region_1 = {'S1_01':[0.020, 0.025],
              'S1_03':[0.020, 0.025],
              'S1_05':[0.020, 0.025],
              'S2_05':[0.024, 0.032],
              'S2_17':[0.020, 0.025],
              'S2_26':[0.030, 0.040],}

y_region_1 = {'S1_01':[2.3, 3.0],
              'S1_03':[2.3, 3.0],
              'S1_05':[2.3, 3.0],
              'S2_05':[1.8, 2.4],
              'S2_17':[2.3, 3.0],
              'S2_26':[2.4, 3.4]}


x_region_2 = {'S1_01':[0.120, 0.150],
              'S1_03':[0.120, 0.150],
              'S1_05':[0.120, 0.150],
              'S2_05':[0.007, 0.008],
              'S2_17':[0.010, 0.014],
              'S2_26':[0.013, 0.019],}   
y_region_2 = {'S1_01':[0.4, 0.9],
              'S1_03':[0.4, 0.9],
              'S1_05':[0.4, 0.9], 
              'S2_05':[5.8, 6.2],
              'S2_17':[6.0, 8.0],
              'S2_26':[8.6, 9.4],}

x_region_3 = {'S1_01':[0.12, 0.13],
              'S1_03':[0.12, 0.13],
              'S1_05':[0.12, 0.13]}
y_region_3 = {'S1_01':[2.00, 2.40],
              'S1_03':[2.00, 2.40],
              'S1_05':[2.00, 2.40]}

x_region_4 = {'S1_01':[0.012, 0.014],
              'S1_03':[0.012, 0.014],
              'S1_05':[0.012, 0.014]}
y_region_4 = {'S1_01':[5.000, 5.600],
              'S1_03':[5.000, 5.600],
              'S1_05':[5.000, 5.600]}

# Create nested dictionary to loop over
x_regions = {'region_1':x_region_1,
             'region_2':x_region_2,
             'region_3':x_region_3, 
             'region_4':x_region_4}
y_regions = {'region_1':y_region_1,
             'region_2':y_region_2,
             'region_3':y_region_3,
             'region_4':y_region_4}
colors = {'region_1':'C0', 'region_2':'C1', 'region_3':'C2', 'region_4':'C3'}


for sx, pulses in pulse_dict.items():
    if sx not in dfs.get_dictionaries('S1'): continue
    if int(sx[3:]) < 16: 
        rec_len = 64
        factor = 1E6

    else: 
        rec_len = 56
        factor = 1E5

    board, _ = dfs.get_board_name(sx)
    
    # Get bias level
    bias_level = dfs.get_bias_level(board = board, shot_number = shot_number)
    
    # Baseline reduction
    pulses = dfs.baseline_reduction(pulses)
    
    # First cleanup
    pulses, bad_indices = dfs.cleanup(pulses, dx = 1, detector_name = sx, 
                         bias_level = bias_level)
        
    # Choose gates, remove bad indices from cleanup
    total = np.delete(total_gates[sx], bad_indices)
    ratio = np.delete(ratios[sx], bad_indices)
    if int(sx[3:]) < 16: 
        threshold = 16 - np.delete(thresholds[sx], bad_indices)
    else:
        threshold = np.zeros(len(pulses))
        
    # Plot 2D histogram
    fig, ax = plt.subplots(2, 2)
#    gs = gridspec.GridSpec(2, 4)
#    ax = np.array([[gs[0, :2], gs[0, 2:]], [None, gs[1, 1:3]]])
    y_bins = np.arange(-20, 20, 0.1)
    x_bins = np.arange(-5, 20, 0.001)
    my_cmap = plt.cm.jet
    my_cmap.set_under('w', 1)
    hist = ax[0, 1].hist2d(total, ratio, bins = [x_bins, y_bins], cmap = my_cmap, vmin = 1)
    ax[0, 1].set_xlabel('$q_{short} + q_{long}$')
    ax[0, 1].set_ylabel('$q_{short}$/$q_{long}$')
    ax[0, 1].set_ylim([-10, 35])
    ax[0, 1].set_xlim([-0.05, 1])
    ax[0, 1].set_title(sx)
    fig.colorbar(hist[3], ax = ax[0, 1])
    fig.set_figheight(8)
    fig.set_figwidth(16)


    for x_key, y_key in zip(x_regions, y_regions):
        if sx not in x_regions[x_key]: continue
        
        # Plot cross from region which we are selecting pulses
        ax[0, 1].plot(np.mean(x_regions[x_key][sx]), np.mean(y_regions[x_key][sx]), color = colors[x_key], marker = 'x', markersize = 10)
        
        # Select random pulses within that region
        indices = np.where((ratio > y_regions[y_key][sx][0]) & 
                           (ratio < y_regions[y_key][sx][1]) & 
                           (total > x_regions[x_key][sx][0]) & 
                           (total < x_regions[x_key][sx][1]))[0]
        
        # Choose and plot random pulses 
        n_pulses = 25
        random_indices = np.random.choice(indices, size = n_pulses)
        random_pulses = pulses[random_indices]
        x_axis = np.tile(np.arange(0, rec_len), (n_pulses, 1)) + threshold[random_indices][:, np.newaxis]
        ax[1, 1].plot(np.transpose(x_axis), np.transpose(random_pulses), color = colors[x_key])
    
plt.show()
    
    
    
    
    
    
    

    
