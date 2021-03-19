#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:34:20 2021

@author: beriksso
"""

'''
Tests sTOF function with a given data set to see if it provides the expected 
result.
'''

import sys
sys.path.insert(0, '../../functions/')
import tofu_functions as dfs
import numpy as np

def test_1():
    # Create a data set to test the function on
    S1 = np.array([1.0, 1.2, 1.4, 3.5, 10, 14, 20, 41, 42, 43, 44, 70, 100, 200, 300]) # S1 time stamps
    S2 = np.array([-5, 0.9, 1.1, 15, 45.1, 102, 250, 299, 300, 301, 350, 360, 370])
    
    '''
    Expected result from this when using a +/- 3 ns time window is
    tof = [-5: nothing,
           0.9: -0.1, -0.3, -0.5, -2.6,
           1.1: 0.1, -0.1, -0.3, -2.4,
           15: 1, 
           45.1: 2.1, 1.1, 
           102: 2,
           250: nothing,
           299: -1, 
           300: 0,
           301: 1,
           350: nothing]

    '''
    
    tof, inds = dfs.sTOF4(S1_times = S1, S2_times = S2, t_back = 3, t_forward = 3, return_indices = True)
    return tof, inds

def test_2():
    # Create a data set to test the function on
    S1 = np.arange(-100, 100)
    S2 = np.arange(-100, 100)
    
    '''
    Expected result from this when using a +/- 0.5 ns time window is 200 zeros.
    '''
    
    tof, inds = dfs.sTOF4(S1_times = S1, S2_times = S2, t_back = 0.5, t_forward = 0.5, return_indices = True)
    return tof, inds

def test_3():
    S1 = np.array([4.0, 5.0, 6.0, 7.0, 8.0])
    S2 = np.array([6.0])
    
    '''
    Expected result from this when using a +/- 2 ns time window is tof = [1, 0, -1], 
    events on the window edge are ignored.
    '''
    
    tof, inds = dfs.sTOF4(S1_times = S1, S2_times = S2, t_back = 2.0, t_forward = 2.0, return_indices = True)
    return tof, inds

def test_4():
    # Create a data set to test the function on
    S1 = np.arange(-100, 100)
    S2 = np.arange(-100, 100)
    
    '''
    Expected result from this when using a +/- 3.1 ns time window is 
    tof = [-100: 0, -1, -2, -3
           -99 : 1,  0, -1, -2, -3
           -98 : 2,  1,  0, -1, -2, -3,
           -97 : 3,  2,  1,  0, -1, -2, -3,
           -96 : 3,  2,  1,  0, -1, -2, -3,
           .
           .
           .
           97 : 3,  2,  1,  0, -1, -2, -3,
           98 : 2,  1,  0, -1, -2, -3,
           99 : 1,  0, -1, -2, -3,
           100: 0, -1, -2, -3 ]
    '''
    
    tof, inds = dfs.sTOF4(S1_times = S1, S2_times = S2, t_back = 4.1, t_forward = 4.1, return_indices = True)
    return tof, inds



    
tof1, inds1 = test_1()
#tof2, inds2 = test_2()
#tof3, inds3 = test_3()
tof4, inds4 = test_4()











