#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:44:32 2021

@author: beriksso
"""

import sys
sys.path.insert(0, '../functions/')
import tofu_functions as dfs
import pickle
import numpy as np

shot_number = 97002

detectors = ['S1_01', 'S1_03', 'S1_05', 'S2_05', 'S2_17', 'S2_26']

pulses = {}
time_stamps = {}
threshold = {}
area = {}
long_plus_short = {}
short_over_long = {}
for detector in detectors:
    board, channel = dfs.get_board_name(detector)
    pulses[detector] = dfs.get_pulses(shot_number = shot_number, board = board, channel = channel)
    time_stamps[detector] = dfs.get_times(shot_number = shot_number, board = board, channel = channel)

    # Find threshold to move pulses to correct trigger threshold
    if int(board) < 6:
        trig_level = dfs.get_trigger_level(board, channel, shot_number)
        threshold[detector] = dfs.find_threshold(pulses[detector], trig_level)
        rec_len = 64
        factor = 1E6

    else:
        rec_len = 56
        factor = 1E5
        
    # Perform baseline reduction
    pulses_bl = dfs.baseline_reduction(pulses[detector])
    
    # Calculate gates
    short = -np.sum(pulses_bl[:, 10:30], axis = 1)
    long  = -np.sum(pulses_bl[:, 30:], axis = 1)
    long_plus_short[detector] = (short + long) / factor
    short_over_long[detector] = short/long    
    
    # Perform sinc interpolation
    x_axis = np.arange(0, rec_len)
    ux_axis = np.arange(0, rec_len, 0.1)
    pulses_sinc = dfs.sinc_interpolation(pulses_bl, x_axis, ux_axis)
    
    # Calculate area under pulse
    area[detector] = -np.trapz(pulses_sinc, dx = 0.1, axis = 1)

    
    
to_pickle = {'pulses':pulses,
             'long_plus_short':long_plus_short,
             'short_over_long':short_over_long,
             'areas':area,
             'thresholds':threshold,
             'shot_number':shot_number,
             'detectors':detectors} 

with open('/common/scratch/beriksso/TOFu/optimization/pulses.pickle', 'wb') as handle:
    pickle.dump(to_pickle, handle, protocol = pickle.HIGHEST_PROTOCOL)

with open('/common/scratch/beriksso/TOFu/optimization/time_stamps.pickle', 'wb') as handle:
    pickle.dump(time_stamps, handle, protocol = pickle.HIGHEST_PROTOCOL)









