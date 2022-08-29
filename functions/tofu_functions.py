#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 08:27:51 2019

@author: beriksso
"""

try: import getdat as gd
except: pass
try: import ppf
except: pass
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
import scipy.constants as constant
import scipy.optimize as optimize
from matplotlib.lines import Line2D
import os

def get_pulses(shot_number, board = 'N/A', channel = 'N/A', detector_name = 'N/A', pulse_start = -1, pulse_end = -1, timer = False):
    '''
    Returns pulse data for a given board, channel and shot number.
    
    Parameters
    ----------
    shot_number : int or string
                JET pulse number.
    board : int or string, optional
          Board number (between 1-10) for requested data. Must be given if
          detector_name is not.
    channel : string, optional
            Channel name for requested data (A, B, C or D). Must be given if 
            detector_name is not.
    detector_name : string, optional
                  Detector name for requested data. Must be given if board and
                  channel are not. E.g. "S1_01", "S1_02", "S2_01", "S2_02" etc.
    pulse_start : int, optional
                Index before which pulses are not returned.
    pulse_end : int, optional
              Index after which pulses are not returned.
    timer : bool, optional
          If set to True, prints the time to execute the function.
    
    Returns
    -------
    pulses : ndarray
           2D array of pulse waveforms where each row corresponds to one pulse. 
           Typically 64 samples in each row for ADQ14 (board 1-5) and 56
           samples for ADQ412 (board 6-10).
           
    Examples
    --------
    >>> get_pulses(94206, 4, 'A')
    array([[29996, 29978, 30005, ..., 29911, 29938, 29929],
           ...,
           [29978, 29969, 29955, ..., 29884, 29920, 29920]], dtype=int16)   
    >>> get_pulses(94206, detector_name = 'S1_04')
    array([[29996, 29978, 30005, ..., 29911, 29938, 29929],
           ...,
           [29978, 29969, 29955, ..., 29884, 29920, 29920]], dtype=int16)   
    '''
    
    if timer: t_start = elapsed_time()
    if detector_name != 'N/A': board, channel = get_board_name(detector_name)
    if int(board) < 10: board = f'0{int(board)}'
    
    file_name = f'M11D-B{board}<DT{channel}'
    
    # Get record lenght
    record_length = get_record_length(board, shot_number)
    
    # Get some of the data or all of it
    if (pulse_start != -1) & (pulse_end != -1):
        pulse_data, _, _ = gd.getbytex(file_name, 
                                       shot_number, 
                                       nbytes = (pulse_end-pulse_start)*record_length*2, 
                                       start = 6+2*record_length*pulse_start, 
                                       order = 12)
    else:
        pulse_data, _, _ = gd.getbyte(file_name, shot_number)
    pulse_data.dtype = np.int16

    # Reshape pulse data
    if len(pulse_data) % record_length != 0: 
        raise Exception(f'Error: Number of records could not be calculated for record length of {record_length} samples.')

    if timer: elapsed_time(t_start, 'get_pulses()')
    
    pulses = np.reshape(pulse_data, [int(len(pulse_data) / record_length), record_length])
    return pulses
       
def get_times(shot_number, board = 'N/A', channel = 'N/A', detector_name = 'N/A', timer = False):
    '''
    Returns trigger time stamps for pulses on given board, channel and shot 
    number in nanoseconds since board initialization.
    
    Parameters
    ----------
    shot_number : int or string
                JET pulse number.
    board : int or string, optional
          Board number (between 1-10) for requested data. Must be given if
          detector_name is not.
    channel : string, optional
            Channel name for requested data (A, B, C or D). Must be given if 
            detector_name is not.
    detector_name : string, optional
                  Detector name for requested data. Must be given if board and
                  channel are not. E.g. "S1_01", "S1_02", "S2_01", "S2_02" etc.
    timer : bool, optional
          If set to True, prints the time to execute the function.
    
    Returns
    -------
    time_stamps : ndarray
                1D array of time stamps.
           
    Examples
    --------
    >>> get_times(97100, 2, 'B')
    array([3.68841388e+06, 9.55251888e+06, 1.62414959e+07, ...,
           1.22809128e+11, 1.22824604e+11, 1.22827203e+11])
    >>> get_times(97100, detector_name = 'S2_04')
    array([3.68841388e+06, 9.55251888e+06, 1.62414959e+07, ...,
           1.22809128e+11, 1.22824604e+11, 1.22827203e+11])
    '''
    

    if timer: t_start = elapsed_time()
    if detector_name != 'N/A': board, channel = get_board_name(detector_name)
    
    if int(board) < 10: board = f'0{int(board)}'
    file_name = f'M11D-B{board}<TM{channel}'
    

    # For ADQ14 time stamps are multiplied by 0.125 to return in ns
    # For ADQ412 time stamps are multiplied by 0.5 to return in ns
    if   int(board) <= 5: mult_factor = 0.125
    elif int(board) > 5: mult_factor = 0.5
    else:
        print('Wrong function call. Example call: data = get_pulses(\'04\', \'A\', 94206)')
        return 0
    
    # Get time stamps
    time_stamps = gd.getbyte(file_name, shot_number)[0]
    if time_stamps.size == 0: return -1
    time_stamps.dtype = np.uint64
    
    if timer: elapsed_time(t_start, 'get_times()')
    return time_stamps * mult_factor
def get_offset(board, shot_number, timer = False):
    '''
    Returns the time from board initialization to JET PRE for given board and 
    shot number in nanoseconds. Required to align the time stamp trains for 
    ADQ412 boards.

    Parameters
    ----------
    board : int or string
          Board number (between 6-10) for requested time offset.
    shot_number : int or string
                JET pulse number.
    timer : bool, optional
          If set to True, prints the time to execute the function.
          
    Returns
    -------
    offset : int
           Nanoseconds between board initialization and JET PRE.
    
    Examples
    --------
    >>> get_offset(6, 97100)
    144185503141
    >>> get_offset(10, 94800)
    92176638652
    '''
    if timer: t_start = elapsed_time()
    if int(board) < 10: board = f'0{int(board)}'
        
    
    file_name = f'M11D-B{board}<OFF' 
    if int(board) < 6:
        raise Exception('Offsets are only available for the ADQ412 cards (i.e. boards 6, 7, 8, 9 and 10)')
    
    # Get offset
    offset = gd.getbyte(file_name, shot_number)[0]
    offset.dtype = np.uint64
    
    if timer: elapsed_time(t_start, 'get_offset()')
    if len(offset) == 0: 
        raise Exception('get_offset() failed to retrieve offset value.')
    return offset[0]
    
def get_temperatures(board, shot_number, timer = False):
    '''
    Returns temperatures (deg C) at different locations on the ADQ412 and ADQ14 boards before and after acquisition.
    For ADQ412 five temperatures are returned, two of which are always 256, these temperature locations on the boards
    are not available for our cards. For ADQ14 seven temperatures are returned.
    For information on where the temperatures are measured see the function GetTemperature() in the ADQAPI manual.
    
    Parameters
    ----------
    board : int or string
          Board number (between 1-10) for requested temperatures.
    shot_number : int or string
                JET pulse number
    timer : bool, optional
          If set to True, prints the time to execute the function.
         
    Returns
    -------
    T0, T1 : ndarray
           Array of temperatures on the boards before and after acquisition.
           
    Examples
    --------
    >>> get_temperatures(4, 94207)
    (array([ 36.125, 256.   , 256.   ,  68.625,  41.5  ], dtype=float32),
     array([ 36.25 , 256.   , 256.   ,  69.5  ,  41.875], dtype=float32)
    '''
    if timer: t_start = elapsed_time()
    
    if int(board) < 10: board = f'0{int(board)}'
    
    file_name_1 = f'M11D-B{board}<T0'
    file_name_2 = f'M11D-B{board}<TE'

    # Get temperatures
    T0 = gd.getbyte(file_name_1, shot_number)[0]
    TE = gd.getbyte(file_name_2, shot_number)[0]
    T0.dtype = np.float32
    TE.dtype = np.float32
    
    if timer: elapsed_time(t_start, 'get_temperatures()')
    return T0, TE
  
def get_time_limit(board, shot_number, timer = False):
    '''
    Returns the acquisition time limit in seconds for a given board and shot
    number.
    
    Parameters
    ----------
    board : int or string
          Board number (between 1-10) for requested trigger level.
    shot_number : int or string
                JET pulse number.
    timer : bool, optional
          If set to True, prints the time to execute the function.
    
    Returns
    -------
    time_limit : int
               Time limit in seconds.
                  
    Examples
    --------
    >>> get_time_limit(1, 97100)
    120
    >>> get_time_limit(6, 97100)
    120
    '''
    if timer: t_start = elapsed_time()

    if int(board) < 10: board = f'0{int(board)}'
    file_name = f'M11D-B{board}>TLM'
    
    # Get trigger level
    tlim, _, _ = gd.getbyte(file_name, shot_number)
    tlim.dtype = np.int16
    time_limit = tlim.byteswap()[0]
    if timer: elapsed_time(t_start, 'get_time_limit()')
    return time_limit

def get_record_length(board, shot_number, timer = False):
    '''
    Returns the record length in number of samples for a given board and shot
    number.
    
    Parameters
    ----------
    board : int or string
          Board number (between 1-10) for requested trigger level.
    shot_number : int or string
                JET pulse number.
    timer : bool, optional
          If set to True, prints the time to execute the function.
    
    Returns
    -------
    record_length : int
                  Record length in number of samples.
                  
    Examples
    --------
    >>> get_record_length(1, 'A', 97100)
    64
    >>> get_record_length(6, 'B', 97100)
    56
    '''
    
    if timer: t_start = elapsed_time()
    if int(board) < 10: board = f'0{int(board)}'
    file_name = f'M11D-B{board}>RLN'
    
    # Get record length
    rln, _, _ = gd.getbyte(file_name, shot_number)
    rln.dtype = np.int16
    record_length = rln.byteswap()[0]
    if timer: elapsed_time(t_start, 'get_record_length()')
    
    # Remove 8 samples of header info for ADQ412
    if int(board)>5: return record_length-8
    else: return record_length


def get_pre_trigger(board, shot_number, timer = False):
    '''
    Returns the number of pre-trigger samples used for the given shot and 
    board.
    
    Parameters
    ----------
    board : int or string
          Board number (between 1-10).
    shot_number : int or string
                JET pulse number.
    timer : bool, optional
          If set to True, prints the time to execute the function.
    
    Returns
    -------
    prt_samples : int
                Number of pre-trigger samples.
    
    Examples
    --------
    >>> get_pre_trigger(1, 97100)
    16
    >>> get_pre_trigger(10, 97200)
    16
    '''
    if timer: t_start = elapsed_time()
    if int(board) < 10: board = f'0{int(board)}'

    file_name = f'M11D-B{board}>PRT'
    
    # Get number of pre trigger samples
    prt, _, _ = gd.getbyte(file_name, shot_number)
    prt.dtype = np.int16
    prt_samples = prt.byteswap()[0]
    if timer: elapsed_time(t_start, 'get_pre_trigger()')
    return prt_samples

def get_bias_level(board, shot_number, timer = False):
    '''
    Returns the bias level for the given shot, board and channel.
    
    Parameters
    ----------
    board : int or string
          Board number (between 1-10).
    shot_number : int or string
                JET pulse number.
    timer : bool, optional
          If set to True, prints the time to execute the function.
    
    Returns
    -------
    blvl : int
         Bias level in codes
    
    Examples
    --------
    >>> get_bias_level(1, 97100)
    27000
    >>> get_bias_level(6, 97100)
    1600
    '''    

    if timer: t_start = elapsed_time()
    
    if int(board) < 10: board = f'0{int(board)}'
    file_name = f'M11D-B{board}>BSL'
    
    # Get bias level
    blvl, _, _ = gd.getbyte(file_name, shot_number)
    blvl.dtype = np.int16    
    
    if timer: elapsed_time(t_start, 'get_bias_level()')
    return blvl.byteswap()[0]

def get_trigger_level(board, channel, shot_number, timer = False):
    '''
    Returns the trigger level in codes used for the given shot, board and 
    channel.
    
    Parameters
    ----------
    board : int or string
          Board number (between 1-10) for requested trigger level.
    channel : string
            Channel name for requested data (A, B, C or D).
    shot_number : int or string
                JET pulse number.
    timer : bool, optional
          If set to True, prints the time to execute the function.
    
    Returns
    -------
    trigger_level : int
                  Trigger level in codes.
                  
    Examples
    --------
    >>> get_trigger_level(8, 'A', 97100)
    1500
    >>> get_trigger_level(1, 'B', 97200)
    26000
    '''
    if timer: t_start = elapsed_time()
    
    if int(board) < 10: board = f'0{int(board)}'
    file_name = f'M11D-B{board}>TL{channel}'
    
    # Get trigger level
    tlvl, _, _ = gd.getbyte(file_name, shot_number)
    tlvl.dtype = np.int16
    trigger_level = tlvl.byteswap()[0]
    if timer: elapsed_time(t_start, 'get_trigger_level()')
    return trigger_level

def baseline_reduction(pulse_data, timer = False):
    '''
    Subtracts the baseline average of the first 10 samples from each pulse. 
    Returns the same pulse data array with the base line centred around zero.
    
    Parameters
    ----------
    pulse_data : ndarray
               2D array of pulse waveforms where each row corresponds to one 
               pulse. Typically 64 samples in each row for ADQ14 (board 1-5) 
               and 56 samples for ADQ412 (board 6-10).
    timer : bool, optional
          If set to True, prints the time to execute the function.
          
    Returns
    -------
    pulses_baseline : ndarray
                    Same as pulse_data but with baseline centred around 0.
                    
    Examples
    --------
    >>> baseline_reduction(
                    array([[26969, 26974, 26992, ..., 26793, 26734, 26671],
                           ...,
                           [26969, 26951, 26974, ..., 26684, 26675, 26630]])
                          )
    array([[  -9.6,   -4.6,   13.4, ..., -185.6, -244.6, -307.6],
           ...,
           [   1.8,  -16.2,    6.8, ..., -283.2, -292.2, -337.2]])
    '''
    
    if timer: t_start = elapsed_time()
    
    # Calculate the average baseline from ten first samples in each record
    baseline = pulse_data[:, :10]
    baseline_av = np.mean(baseline, axis = 1)
    
    # Create array of baseline averages with same size as pulse_data
    baseline_av = np.reshape(baseline_av, (len(baseline_av), 1))
    baseline_av = np.repeat(baseline_av, np.shape(pulse_data)[1], axis = 1)
    pulses_baseline = pulse_data - baseline_av
    if timer: elapsed_time(t_start, 'baseline_reduction()')
    return pulses_baseline

def remove_led(time_stamps, timer = False):
    '''
    Removes chunk of LED data at the end of time stamp train. Assumes that the
    frequency of the LED source is 5 kHz.
    
    Parameters
    ----------
    time_stamps : ndarray
                1D array of time stamps.
    timer : bool, optional
          If set to True, prints the time to execute the function.
          
    Returns
    -------
    t : ndarray
      1D array of time stamps without LED chunk.
    led_start : int
              Index in time_stamps at which LED started.
    '''
    
    if timer: t_start = elapsed_time()
    
    if len(time_stamps) == 0:
        print('Time stamp array is empty.')
        return 0, 0
    
    # Find time difference between each time stamp
    dT = np.diff(time_stamps)
   
    dT_arg = np.where((dT > 190000) & (dT < 200000))[0]
    
    combo_counter = 0
    A = 0
    for led_start in dT_arg:
        
        B = led_start - A
        if B == 1: combo_counter += 1
        else: combo_counter = 0
        A = led_start    
        
        if combo_counter == 10: break
    
    if timer: elapsed_time(t_start, 'remove_led()')

    if combo_counter < 10: 
        print('LED\'s not found, returning full data set.')
        return time_stamps

    led_start -= 10
    t = time_stamps[0:led_start - 10]
    return t, led_start - 10

def find_threshold(pulse_data, trig_level, timer = False):
    '''
    Finds the point in the pulse which crosses the trigger level 
    (generally sample 16, 17, 18 or 19 ns for ADQ14). Mainly relevant for ADQ14
    cards since the number of pre trigger samples varies.
    
    Parameters
    ----------
    pulse_data : ndarray
               2D array of pulse waveforms where each row corresponds to one 
               pulse. Typically 64 samples in each row for ADQ14 (board 1-5) 
               and 56 samples for ADQ412 (board 6-10).
    trig_level : int
               Trigger level used during acquisition in codes.
    timer : bool, optional
          If set to True, prints the time to execute the function.
    
    Returns
    -------
    thr_crossing : ndarray
                 1D array of indices at which each pulse crosses the given
                 threshold.
    
    Examples
    --------
    >>> find_threshold(pulse_data, 26000)
    array([16, 16, 16, ..., 16, 18, 16])
    '''
    
    if timer: t_start = elapsed_time()
    # Subtract the trigger level from pulse data
    pulse_data = pulse_data - trig_level

    # Find all negative numbers (positive numbers correspond to elements above the threshold)
    neg_pulse_data = np.where(pulse_data <= 0)

    # Find the index of the first ocurrence of each number in neg_pulse_data
    # Example: neg_pulse_data[0] = [0(this one), 0, 0, 0, 0, 1(this one), 1, 1, 2(this one), 2, 2, 2...]
    u, indices = np.unique(neg_pulse_data[0], return_index = True)

    # Choose the corresponding elements from neg_pulse_data[1]
    thr_crossing = neg_pulse_data[1][indices]
    if timer: elapsed_time(t_start, 'find_threshold()')

    return thr_crossing

def sinc_interpolation(pulse_data, x_values, ux_values, timer = False):
    '''
    Returns since-interpolation of given pulse data set.
    See Matlab example: 
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html
    
    Parameters
    ----------
    pulse_data : ndarray
               2D array of pulse waveforms where each row corresponds to one 
               pulse. Typically 64 samples in each row for ADQ14 (board 1-5) 
               and 56 samples for ADQ412 (board 6-10). NOTE: pulse_data must be
               baseline reduced (see baseline_reduction() function).
    x_values : ndarray
             1D array of values corresponding to x_axis of pulse_data. 
             Length between each point in x_values must be constant.
    ux_values: ndarray 
             1D array similar to x_values but upsampled. Length between each
             point in ux_values must be constant.
    timer : bool, optional
          If set to True, prints the time to execute the function.
          
    Returns
    -------
    u_pulse_data : ndarray
                 2D array of sinc-interpolated pulse waveforms where each row
                 corresponds to one pulse.
        
    Examples
    --------
    >>> # Let n be the number of records and m the number of samples per record
    >>> x_values = np.arange(0, n)
    >>> ux_values = np.arange(0, n, 0.1) # Upsampled 10 times
    >>> u_pulse_data = since_interpolation(pulse_data, x_values, ux_values)    
    '''
    if timer: t_start = elapsed_time()
    
    # Record length
    length = np.shape(pulse_data)[1]
    n_records = len(pulse_data)
    
    # Store results here
    u_pulse_data = np.zeros([len(pulse_data), len(ux_values)])
    counter = 0
    
    if length != len(x_values):
        print('pulse_data and x_values must be the same length.')
        return 0
    
    # Chunk data if too many records
    if n_records > 1E+6:
        # Chunk array in chunks of ~1E6 rows
        n_chunks = int(np.ceil(len(pulse_data) / 1E+6))
        chunked_data = np.array_split(pulse_data, n_chunks, axis = 0)
    # Otherwise use full data set at once
    else: chunked_data = [pulse_data]
    
    # Do sinc interpolation for each chunk of data
    for pulse_data in chunked_data:
        # Find period of x_values
        period = x_values[1] - x_values[0]    
        
        # Set up sinc matrix
        sinc_matrix = np.tile(ux_values, (len(x_values), 1)) - np.tile(x_values[:, np.newaxis], (1, len(ux_values)))
        
        # Perform sinc interpolation
        sinc = np.sinc(sinc_matrix / period)
        u_pulse_data[counter:len(pulse_data) + counter, :] = np.dot(pulse_data, sinc)
        counter += len(pulse_data)
    
    if timer: elapsed_time(t_start, 'sinc_interpolation()')
    return u_pulse_data
    
def time_pickoff_CFD(pulse_data, fraction = 0.3, timer = False):
    '''
    Returns the times of arrival for a 2D array of pulses using a constant
    fraction and a linear interpolation method.
    
    Parameters
    ----------
    pulse_data : ndarray,
               2D array of pulse waveforms where each row corresponds to one 
               pulse. Typically 64 samples in each row for ADQ14 (board 1-5) 
               and 56 samples for ADQ412 (board 6-10). NOTE: pulse_data must be
               baseline reduced (see baseline_reduction function).
    fraction : float
             Fraction at which to perform the linear interpolation
    timer : bool, optional
          If set to True, prints the time to execute the function.
          
    Returns
    -------
    new_time : ndarray
             1D array of times-of-arrival for each pulse.

    Examples
    --------
    >>> time_pickoff_CFD(pulse_data, fraction = 0.05)
    [12.5149318,  14.97846766, ... 12.53181554, 12.39070941, 12.94160379]
    '''
    
    new_time = np.zeros([len(pulse_data)])

    # Determine whether data needs to be chunked or not
    if len(pulse_data) > 5E+5: chunk_data = True
    else: chunk_data = False

    if timer: t_start = elapsed_time()
    if chunk_data:
        # Chunk array in chunks of ~5E5 rows
        n_chunks = int(np.ceil(len(pulse_data) / 5E+5))
        chunked_data = np.array_split(pulse_data, n_chunks, axis = 0)
        

    else: chunked_data = [pulse_data]
    new_time_counter = 0
    for pulse_data in chunked_data:
        # Find the minima and a fraction of the minima
        minima = np.min(pulse_data, axis = 1)
        minima_fraction = minima * fraction
        # Find position of minimum
    #    minima_pos = np.argmin(pulse_data, axis = 1)
    #    print('Warning: ' + str(len(minima_pos[minima_pos < 100])) + ' pulses have minimum before 10 ns.')
        
    
        # Find the index of the point closest to the fraction of the minimum
        # Look only in the first 25 ns (leading edge) of the pulse
        x_closest = find_points(pulse_data[:, 0:250], minima_fraction, timer = timer)
    
    
        # Set up for simple linear regression
        reg_x = np.zeros([len(x_closest), 3])
        reg_y = np.zeros([len(x_closest), 3])
        array_1D = np.arange(0, len(pulse_data), 1)
        
        # Choose the three points on which to perform simple linear regression
        reg_y[:, 0] = pulse_data[array_1D, x_closest - 1]
        reg_y[:, 1] = pulse_data[array_1D, x_closest]
        reg_y[:, 2] = pulse_data[array_1D, x_closest + 1]
    
        reg_x[:, 0] = x_closest - 1
        reg_x[:, 1] = x_closest
        reg_x[:, 2] = x_closest + 1
        
        # Perform simple linear regression
        slopes, intercepts = linear_regression(reg_x, reg_y, timer = timer)
        # Solve the y = kx + m equation for x. y = minima_fraction
        new_time[new_time_counter:len(pulse_data)+new_time_counter] = (minima_fraction - intercepts) / slopes
        new_time_counter += len(pulse_data)
        

    if timer: elapsed_time(t_start, 'time_pickoff_CFD()')
    return new_time


 
def linear_regression(x_data, y_data, timer = False):
    '''
    Returns the slope (A) and intersection (B) for a simple linear regression 
    on x and y data on the for y = Ax+B.
    
    Parameters
    ----------
    x_data : ndarray,
            2D array of x-data        
    y_data : ndarray,
            2D array of y-data
    timer : bool, optional
          If set to True, prints the time to execute the function.
          
    Returns
    -------
    slope : ndarray,
          1D array of slopes (A) of the linear regression on the form y = Ax+B
          for each row of x_data and y_data
    intercept : ndarray,
          1D array of intercepts (B) of the linear regression on the form 
          y = Ax+B for each row of x_data and y_data
    
    Examples
    --------
    >>> x_data = np.random.randint(0, 100, (10, 50))
    >>> y_data = np.random.randint(0, 100, (10, 50))
    >>> linear_regression(x_data, y_data)
    (array([-0.12838402,  0.03837328,  0.23679145, -0.11515894, -0.08433706,
            -0.01809531, -0.02958148, -0.01858007,  0.13560102,  0.30361892]),
     array([53.50304412, 49.65230379, 37.04726166, 53.7653137 , 52.42589113,
            49.156337  , 48.59527858, 48.24523488, 38.27435695, 37.49710363]))    
    '''
    if timer: t_start = elapsed_time()
    
    # Find average
    x_mean = np.mean(x_data, axis = 1)
    y_mean = np.mean(y_data, axis = 1)
    
    # product_1-3 correspond to the three products for calculating beta in 
    # https://en.wikipedia.org/wiki/Simple_linear_regression
    product_1 = np.transpose(np.transpose(x_data) - x_mean)
    product_2 = np.transpose(np.transpose(y_data) - y_mean)
    product_3 = product_1 ** 2
    
    # Calculate slopes and intersection (y = slope*x + intercept)
    slope = np.sum(product_1 * product_2, axis = 1) / np.sum(product_3, axis = 1)
    intercept = np.mean(y_data, axis = 1) - slope * x_mean
    
    if timer: elapsed_time(t_start, 'linear_regression()')    
    return slope, intercept


def find_points(pulse_data, value, timer = False):
    '''
    Returns the indicies of the points closest to "value" in pulse_data.
    
    Parameters
    ----------
    pulse_data : ndarray,
               2D array of pulse height data where each row corresponds to one 
               record. NOTE: pulse_data must be baseline reduced (see 
               baseline_reduction() function).
    value : ndarray,
            1D array of values
    timer : bool, optional
          If set to True, prints the time to execute the function.
          
    Returns
    -------
    index : ndarray,
          1D array of indices from pulse_data corresponding to the position
          in each row with value closest to "value".
    
    Examples
    --------
    >>> values = np.random.randint(0, 10, len(pulse_data))
    >>> find_points(pulse_data, 16)
    array([16, 16,  15, 16, ..., 16, 15, 17, 15, 16, 16])  
    '''
    
    if timer: t_start = elapsed_time()
    
    # Subtract the constant fraction value from the data set
    delta = pulse_data - value[:, None]
    
    # Find the index of the first positive value
    mask = delta <= 0
    
    index = np.argmax(mask, axis = 1) 
    
    if timer: elapsed_time(t_start, 'find_points()')
    return index   
    
def sTOF4(S1_times, S2_times, t_back, t_forward, return_indices = False, timer = False):
    '''
    Returns the time differences between S1_times and S2_times given a search
    window. We loop over the time stamps in S2, for each S2 time stamp we 
    search for an event in S1 within a given time window for events. If such an
    event is found the time difference is calculated.
    
    Parameters
    ----------
    S1_times : ndarray
             1D array of time stamps for one S1, typically given in ns.
    S2_times : ndarray
             1D array of time stamps for one S2, typically given in ns.
    t_back : int or float
           Time to look backwards in time from any given S2 time stamp. 
           Constitutes together with t_forward, the time window in which we 
           search for S1 events.
    t_forward : int or float
              Time to look forwards in time from any given S2 time stamp. 
              Constitutes together with t_back, the time window in which we 
              search for S1 events.
    return_indices : bool, optional
                   If set to true, returns S1 and S2 indices for each found
                   coincidence.
    timer : bool, optional
          If set to True, prints the time to execute the function.
                 
    Returns
    -------
    delta_t : ndarray
            1D array of time differences between S1 and S2 time stamps (i.e. 
            times-of-flight).
    indices : ndarray
            2D array of indices for each coincidence. First column
            corresponds to S1, second column to S2.
            
    Examples
    --------
    >>> s1_times = np.array([10, 20, 30, 31, 32, 34])
    >>> s2_times = np.array([9, 15, 33])
    >>> sTOF(s1_times, s2_times, 4, 4)
    array([-1.,  3.,  2.,  1., -1.])
    '''
    
    if timer: t_start = elapsed_time()
    # Define time windows
    w_low = S2_times - t_back
    w_high = S2_times + t_forward
    
    # We will store time differences in dt
    dt = -9999 * np.ones(5 * len(S2_times))
    ind = -9999 * np.ones([5 * len(S2_times), 2])
    counter = 0
    finished = False
    lowest_indices = np.searchsorted(S1_times, w_low)

    for i in range(0, len(S2_times)):
        lowest_index = lowest_indices[i]
        search_sorted = 0
        # Find the time stamp in S1 closest to wLow (rounded up, i.e. just inside the window)
        while True:

            # Increase to next event
            low_index = lowest_index + search_sorted
            
            '''
            Is the lowest time window beyond the final event in S1?
            I.e. S2[i] - t_back > S1[-1]. Then we can stop searching.
            '''
            if lowest_index == len(S1_times): 
                finished = True
                break
            
            '''
            Have we run out of S1 events in our search?
            There may however be more S2 events that produce coincidences 
            with S1[-1]. We go to next S2 event and check if S1[-1] is within
            the next S2 time window.
            '''
            if low_index == len(S1_times):
                break
            
            # If the time stamp in S1 is beyond the window we go to next S2 time (there are no more time stamps within this window)
            if S1_times[low_index] >= w_high[i]: break
            # If the time stamp in S1 is before the window check the next time stamp (should never happen currently)
            if S1_times[low_index] <= w_low[i]: 
                search_sorted += 1
                continue
        
            # If dt is not big enough to fit all events
            if counter == len(dt):
                # Copy arrays
                temp_dt = np.copy(dt)
                temp_ind = np.copy(ind)
                
                # Increase array size by factor 2
                dt = -9999 * np.ones(2 * len(dt))
                ind = -9999 * np.ones([2 * len(ind), 2])
                
                # Fill arrays with values
                dt[:len(temp_dt)] = temp_dt
                ind[:len(temp_ind)] = temp_ind
                
            
            # If there is an event we calculate the time difference
            dt[counter] =  S2_times[i] - S1_times[low_index]
            
            # Save the S1 and S2 index of the event
            ind[counter][0] = low_index
            ind[counter][1] = i
            counter += 1
            search_sorted += 1
            
        if finished: break
    
    # Find and remove all fails from dt
    delta_t = dt[(dt != -9999)]

    ind_S1 = ind[:, 0][ind[:, 0] != -9999]
    ind_S2 = ind[:, 1][ind[:, 1] != -9999]

    if timer: elapsed_time(t_start, 'sTOF4()')
    if return_indices:
        indices = np.array([ind_S1, ind_S2], dtype = 'int')
        return delta_t, indices
    else: return delta_t
    

def get_detector_name(board, channel, timer = False):
    '''
    Returns the detector name corresponding to the given board and channel.
    The order of the S2's on the ADQ412's are back to front
    
    Parameters
    ----------
    board : string or int,
          String or integer corresponding to the board number (1-10)
    channel: string,
           String containing the channel number ('A', 'B', 'C', or 'D')
    timer : bool, optional
          If set to True, prints the time to execute the function.
                 
    Returns
    -------
    detector_name : string,
                  String containing the detector name corresponding to the
                  input board and channel.
            
    Examples
    --------
    >>> get_detector_name(1, 'A')
    'S1_01'
    '''
    
    if timer: t_start = elapsed_time()
    detectors = ['S1_01', 'S2_01', 'S2_02', 'S2_03', 
                 'S1_02', 'S2_04', 'S2_05', 'S2_06',
                 'S1_03', 'S2_07', 'S2_08', 'S2_09',
                 'S1_04', 'S2_10', 'S2_11', 'S2_12',
                 'S1_05', 'S2_13', 'S2_14', 'S2_15',
                 'S2_31', 'S2_32', 'ABS_REF', '1kHz_CLK',
                 'S2_27', 'S2_28', 'S2_29', 'S2_30',
                 'S2_23', 'S2_24', 'S2_25', 'S2_26', 
                 'S2_19', 'S2_20', 'S2_21', 'S2_22',
                 'DEAD', 'S2_16', 'S2_17', 'S2_18']
    cha = np.array(['A', 'B', 'C', 'D'])
    detector_name = detectors[4 * (int(board)-1) + np.where(channel == cha)[0][0]]
    if timer: elapsed_time(t_start, 'get_detector_name()')    
    return detector_name

def get_board_name(detector_name, timer = False):
    '''
    Returns the board and channel for a corresponding detector name. The order 
    of the S2's on the ADQ412's are back to front.
    
    Parameters
    ----------
    detector_name : string,
          String containing the detector name ('S1_01',...,'S1_05' or 
          'S2_01', ..., 'S2_32')
    timer : bool, optional
          If set to True, prints the time to execute the function.
                 
    Returns
    -------
    board : string,
          String containing the board number corresponding to the input 
          detector name.
    channel : string,
            String containing the channel corresponding to the input detector
            name.
            
    Examples
    --------
    >>> get_board_name('S1_01')
    ('01', 'A')
    '''
    
    if timer: t_start = elapsed_time()

    detectors = np.array(['S1_01', 'S2_01', 'S2_02', 'S2_03', 
                 'S1_02', 'S2_04', 'S2_05', 'S2_06',
                 'S1_03', 'S2_07', 'S2_08', 'S2_09',
                 'S1_04', 'S2_10', 'S2_11', 'S2_12',
                 'S1_05', 'S2_13', 'S2_14', 'S2_15',
                 'S2_31', 'S2_32', 'ABS_REF', '1kHz_CLK',
                 'S2_27', 'S2_28', 'S2_29', 'S2_30',
                 'S2_23', 'S2_24', 'S2_25', 'S2_26', 
                 'S2_19', 'S2_20', 'S2_21', 'S2_22',
                 'DEAD', 'S2_16', 'S2_17', 'S2_18'])
    
    channels = np.array(['A', 'B', 'C', 'D'])
    pos = np.where(detectors == detector_name)[0]

    # Find board number
    board = int((np.floor(pos / 4))[0] + 1)
    # Add '0' in front
    if board < 10: board = '0' + str(board)
    else: board = str(board)
    
    # Find channel
    channel = channels[pos % 4][0]
    if timer: elapsed_time(t_start, 'get_board_name()')    
    return board, channel

def get_shifts(shift_file, timer = False):
    '''
    Returns the shifts (written in shift_file) required to line up all S1-S2 
    combinations and shift the gamma peak 3.7 ns. Method is outlined in
    Eriksson, Benjamin, et al. 
    "New method for time alignment and time calibration of the TOFOR 
    time-of-flight neutron spectrometer at JET." 
    Review of Scientific Instruments 92.3 (2021): 033538.

    Parameters
    ----------
    shift_file : string,
               The path to the shift file.
    timer : bool, optional
          If set to True, prints the time to execute the function.
                 
    Returns
    -------
    shifts : dict,
           Dictionary with keys 
           dict_keys(['S1_01', 'S1_02', 'S1_03', 'S1_04', 'S1_05'])
           Each contains 32 shifts (ns) corresponding to the S1 vs. S2 shifts.
            
    Examples
    --------
    >>> shifts = get_shifts()
    >>> shifts.keys()
    dict_keys(['S1_01', 'S1_02', 'S1_03', 'S1_04', 'S1_05']
    >>> shifts['S1_01']
    array([ -18.95286,  -19.35286,..., -118.55286, -117.75286])
    '''

    if timer: t_start = elapsed_time()
    A = np.loadtxt(shift_file, dtype = 'str')
    
    # Get gamma peak shifts for S1-5 vs S2's
    gamma_peak = np.array(A[0:-4, 1], dtype = 'float')
    # Get neutron peak shifts for S1-5 vs S1's
    neutron_peak = np.array(A[-4:, 1], dtype = 'float')
     
    # Gamma peak should be located at 3.7 ns
    g_peak = 3.7
    
    # Dictionary
    shifts = {'S1_01':[], 'S1_02':[], 'S1_03':[], 'S1_04': [], 'S1_05':[]}

    # This gives how much one needs to shift each TOF spectrum in order to line up with the S1_5 vs S2's at 3.7 ns
    shifts['S1_05'] = g_peak - gamma_peak
    shifts['S1_04'] = shifts['S1_05'] - neutron_peak[3]
    shifts['S1_03'] = shifts['S1_05'] - neutron_peak[2]
    shifts['S1_02'] = shifts['S1_05'] - neutron_peak[1]
    shifts['S1_01'] = shifts['S1_05'] - neutron_peak[0]
    
    if timer: elapsed_time(t_start, 'get_shifts()')
    return shifts

def get_pulse_area(pulses, u_factor, timer = False):
    '''
    Returns the areas under the pulses.

    Parameters
    ----------
    pulses : ndarray,
            2D array of pulse waveforms-
    u_factor : int,
             Up-sampling factor, must be set equal to the up-sampling performed
             by the sinc-interpolation.
    timer : bool, optional
          If set to True, prints the time to execute the function.
                 
    Returns
    -------
    pulse_area : ndarray,
               1D darray of pulse areas.
            
    Examples
    --------
    >>> u = 10
    >>> sinc_pulses = sinc_interpolation(pulses, x_values, ux_values, u)
    >>> get_pulse_area(sinc_pulses, u)
    array([ -732.5, -1724., ..., -6221.5, -3151. ])    
    ''' 
    
    if timer: t_start = elapsed_time()
    
    # Chunk data if too many pulses, otherwise we run into memory issues
    pulse_area = np.zeros(len(pulses))
    
    if len(pulses) > 1E+6:
        # Chunk array in chunks of ~1E6 rows
        n_chunks = int(np.ceil(len(pulses) / 1E+6))
        chunked_data = np.array_split(pulses, n_chunks, axis = 0)
    
    # Otherwise use full data set at once
    else: 
        chunked_data = [pulses]
    
    # Find area under pulse
    counter = 0
    for chunk in chunked_data:
        pulse_area[counter:len(chunk) + counter] = np.trapz(chunk, axis = 1, dx = 1. / u_factor)
        counter += len(chunk)
        
    if timer: elapsed_time(t_start, 'get_pulse_area()')
    return pulse_area

def get_energy_calibration(areas, detector_name, timer = False):
    '''
    Returns the deposited energy (MeVee) in the given detector using the energy
    calibration given in the energy calibration folder.
    
    Parameters
    ----------
    areas : ndarray,
          1D array of pulse areas. 
    detector_name : string,
                  Detector name corresponding to the pulse areas being parsed.
    timer : bool, optional
          If set to True, prints the time to execute the function.                 
                    
    Returns
    -------
    energy_array : ndarray,
                 1D array of deposited energies in MeVee.
            
    Examples
    --------
    >>> energies = get_energy_calibration(areas, 'S1_01')
    array([0.00134, 0.23145, ..., 0.02134])
    ''' 
    
    if timer: t_start = elapsed_time()
    
    raise_exception = False
    # Load calibration data for given detector
    if detector_name[0:2] == 'S1':
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../energy_calibration/energy_calibration_S1.txt')
        cal = np.loadtxt(filename, usecols = (0,1))[int(detector_name[3:]) - 1]

        cal_factor = 3000.
    elif detector_name[0:2] == 'S2':
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../energy_calibration/energy_calibration_S2.txt')
        cal = np.loadtxt(filename, usecols = (0,1))[int(detector_name[3:]) - 1]
        
        if int(detector_name[3:]) <= 15: cal_factor = 3000.
        elif int(detector_name[3:]) > 15: cal_factor = 350.
        else: raise_exception = True   
    else: raise_exception = True
    if raise_exception: raise Exception('Please supply the detector type as the second parameter (SX = \'S1_x\' x = [01, 05] or SX = \'S2_x\' x = [01, 32])')        
    
    # Calculate energy from area
    energy_array = (cal[0] + cal[1] * areas / cal_factor ) / 1000.
    if timer: elapsed_time(t_start, 'get_energy_calibration()')
    return energy_array

def find_time_range(shot_number, n_yield=False, timer=False):
    '''
    Returns the time range for which the 99% of the neutron events have
    ocurred using TIN/RNT PPF.
    
    Parameters
    ----------
    shot_number : int,
                Jet puse number.
    n_yield : bool, optional
            If set to True, returns the total neutron yield in the given time 
            range.
    timer : bool, optional
          If set to True, prints the time to execute the function.                 
    
    Returns
    -------
    energy_array : ndarray,
                 1D array of deposited energies in MeVee.
            
    Examples
    --------
    >>> find_time_range(98044)
    array([47.26, 56.17])
    >>> find_time_range(98044, n_yield=True)
    (array([47.26, 56.17]), 5.92e+16)
    ''' 

    if timer: t_start = elapsed_time()

    # Import fission chamber information
    f_chamber = ppf.ppfget(shot_number, dda = "TIN", dtyp = "RNT")
    f_data = f_chamber[2]
    f_times = f_chamber[4]

    if len(f_times) == 0: 
        print('WARNING: Fission chamber data unaviailable.')
        time_slice = np.array([40., 70.])
        neutrons = None
    else:
        # Create cumulative distribution function
        f_cdist = np.cumsum(f_data) / np.sum(f_data)
        
        # Find the time before which 99.5% of the neutron yield occurs
        arg0 = np.searchsorted(f_cdist, 0.005)
        arg1 = np.searchsorted(f_cdist, 0.995)
        
        t0 = f_times[arg0]
        t1 = f_times[arg1]
        time_slice = np.array([t0, t1])
        
        # Calculate neutron yield in time range
        neutrons = f_data[arg0:arg1+1].sum()*0.006
        
    if timer: elapsed_time(t_start, 'find_time_range()')
    if n_yield: return time_slice, neutrons
    return time_slice

def cleanup(pulses, dx, detector_name, bias_level, baseline_cut = np.array([[0, 0], [0, 0]]), timer = False):
    '''
    Takes an array of baseline reduced pulses and removes junk pulses.
    
    Parameters
    ----------
    pulses : ndarray,
           2D array of pulse waveforms where each row corresponds to one 
           waveform. Must be baseline reduced [see baseline_reduction()].
    dx : float,
       Distance between each point on the x-axis. Typically 1 ns.
    bias_level : int,
               Digitizer bias level used for the shot.
    baseline_cut : ndarray,
                 Baselines which fluctuate more than the thresholds given in 
                 baseline_cut (in codes) are removed.
    timer : bool, optional
          If set to True, prints the time to execute the function.                 
                    
    Returns
    -------
    new_pulses : ndarray,
                 2D array of pulse waveforms without junk waveforms.
    bad_indices : ndarray,
                Index numbers of the removed waveforms.
            
    Examples
    --------
    >>> cleanup(pulses, 1, 'S1_01', 27000)
    [[ 2.00e-01 -2.80e+00  2.00e-01 ... -1.98e+01 -1.88e+01 -1.68e+01]
     [ 1.00e+00  2.00e+00  1.00e+00 ... -1.70e+01 -1.40e+01 -1.20e+01]
     [-1.00e+00  2.00e+00 -2.00e+00 ... -1.75e+02 -1.57e+02 -1.38e+02]
     ...
     [ 3.00e-01 -7.00e-01  3.30e+00 ... -1.70e+00  3.00e-01 -7.00e-01]
     [ 3.90e+00 -1.00e-01 -1.10e+00 ... -7.10e+00 -5.10e+00 -8.10e+00]
     [ 8.00e-01  1.80e+00 -1.20e+00 ... -1.02e+01 -1.22e+01 -9.20e+00]]
    [11, 22, 25,..., 1572, 1574]
    ''' 
    
    if timer: t_start = elapsed_time()
    if detector_name not in get_dictionaries('merged').keys():
        raise Exception('Unknown detector name.')
    # Remove anything with a positive area
    area = np.trapz(pulses, axis = 1, dx = dx)
    indices = np.where(area > 0)[0]

    # Remove anything with points on the baseline far from requested baseline 
    if bias_level not in [27000, 30000, 1600]: print('WARNING: The function cleanup() bases it\'s cuts on a bias level of 27k or 30k for ADQ14 and 1.6k codes for ADQ412. This shot has a bias level of ' + str(bias_level) + ' codes.')
    if np.abs(np.mean(pulses[0, 0:10])) > 10: print('WARNING: The function cleanup() requires pulses with a baseline centred around 0.')
    
    '''
    Left hand side baseline
    '''
    # Define ADQ14 and ADQ412 thresholds for the left hand side baseline
    if not np.array_equal(baseline_cut, np.array([[0, 0], [0, 0]])):
        low_threshold = baseline_cut[0][0]
        high_threshold = baseline_cut[0][1]
    elif int(detector_name[3:]) < 16:
        high_threshold = 50
        low_threshold = -50
    else:
        high_threshold = 5
        low_threshold = -5
    
    # Find baselines which violate the thresholds
    baseline_left = pulses[:, 0:11]
    odd_bl_left = np.unique(np.where((baseline_left < low_threshold) | (baseline_left > high_threshold))[0])

    '''
    The following thresholds on the right hand side baseline could be set to
    remove large pulses or pile-up events. It is ignored for now as the
    integration of the pulses is only done between 10-30 ns.
    
    Right hand side baseline
        # Define ADQ14 and ADQ412 thresholds for the right hand side baseline
        if not np.array_equal(baseline_cut, np.array([[0, 0], [0, 0]])):
            low_threshold = baseline_cut[1][0]
            high_threshold = baseline_cut[1][1]
        elif int(detector_name[3:]) < 16:
            high_threshold = 50
            low_threshold = -200
        else:
            high_threshold = 5
            low_threshold = -20
        
        if int(detector_name[3:]) < 6: baseline_right = pulses[:, 40:]
        else: baseline_right = pulses[:, 35:]
        
        odd_bl_right = np.unique(np.where((baseline_right < low_threshold) | (baseline_right > high_threshold))[0])
    '''
    
    odd_bl_right = np.array([], dtype = 'int')
    # Indices for pulses to be removed
    bad_indices = np.append(odd_bl_right, odd_bl_left)
    bad_indices = np.unique(np.append(bad_indices, indices))

    # Remove pulses with odd baseline
    new_pulses = np.delete(pulses, bad_indices, axis = 0)    

    if timer: elapsed_time(t_start, 'cleanup()')
    return new_pulses, bad_indices

def inverted_light_yield(light_yield, function = 'gatu', check = True, timer = False):
    '''
    Takes an array of light yields (MeVee) and converts to proton recoil energy
    (MeV) using the look-up table of the inverted light yield function 
    specified in "function".

    Parameters
    ----------
    light_yield : ndarray,
                1D array of light yields given in MeVee.
    function : str, optional
             Set to "gatu" to use the light yield function from M. Gatu Johnson,
             set to "stevenato" to use light yield function from Stevanato, L., 
             et al. "Light output of EJ228 scintillation neutron detectors." 
             Applied Radiation and Isotopes 69.2 (2011): 369-372.
    timer : bool, optional
          If set to True, prints the time to execute the function.
                 
    Returns
    -------
    proton_recoil : ndarray,
                  Array of proton recoil energies (MeV).
            
    Examples
    --------
    >>> E_ly = [1, 2, 3] # (MeVee)
    >>> inverted_light_yield(E_ly)
    array([3.84391241 6.12264731 8.2078698])    
    '''
    
    if timer: t_start = elapsed_time()
    # Cast into numpy array
#    light_yield = np.array([light_yield])
    
    # Import look-up table
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, f'../inverted_light_yield/look_up_{function}.txt')
    table = np.loadtxt(filename)
    
    # Check if the look-up table matches the light yield function
    if check:
        __proton_recoil = np.arange(0, 3, 0.1)
        __light_yield = light_yield_function(__proton_recoil, function)
        __proton_recoil_r = inverted_light_yield(__light_yield, function, False)
        if not np.allclose(__proton_recoil, __proton_recoil_r, 0.0001):
            msg = 'Inverted light yield look-up table does not ' \
                  'match the light yield function. Please '\
                  'regenerate the look-up table.'
            raise Exception(msg)
    
    
    # Find closest value in look-up table for the light yield
    proton_recoil = np.zeros(np.shape(light_yield))
    for i, ly in enumerate(light_yield): 
        arg = np.searchsorted(table[:, 1], ly)
        proton_recoil[i] = table[arg][0]
    
    if timer: elapsed_time(t_start, 'inverted_light_yield()')
    return proton_recoil
    
def light_yield_function(proton_energy, function = 'gatu', s = 0.73, timer = False):
    '''
    Takes an array of proton recoil energies (MeV) and converts to light yield
    (MeVee) using the light yield function specified in "function".

    Parameters
    ----------
    proton_energy : ndarray,
                  1D array of proton recoil energies given in MeV.
    function : str, optional
             Set to "gatu" to use the light yield function from M. Gatu Johnson,
             set to "stevenato" to use light yield function from Stevanato, L., 
             et al. "Light output of EJ228 scintillation neutron detectors." 
             Applied Radiation and Isotopes 69.2 (2011): 369-372.
    s : float,
      Arbitrary scaling factor, default set to s=0.73, to ensure that the light
      yield function matches the measured spectrum.
    timer : bool, optional
          If set to True, prints the time to execute the function.
                 
    Returns
    -------
    light_yield : ndarray,
                  Array of light yields (MeVee).
            
    Examples
    --------
    >>> E_p = [3.84391241, 6.12264731, 8.2078698] # (MeV)
    >>> light_yield_function(E_p)
    array([1.00004149, 2.00002003, 3.00005272])    
    '''
   
    if timer: t_start = elapsed_time()
    # Cast into numpy array
    proton_energy = np.array(proton_energy)
    
    if function == 'gatu':
        '''
        Light yield function from M. Gatu Johnson's thesis
        '''
        # Different energy ranges
        low_mask    = proton_energy <= 1.9
        medium_mask = (proton_energy > 1.9) & (proton_energy <= 9.3)
        high_mask   = proton_energy > 9.3
        
        a1 = 0.0469
        b1 = 0.1378
        c1 = -0.0183
        
        a2 = -0.01420
        b2 =  0.12920
        c2 =  0.06970
        d2 = -0.00315
        
        a3 = -1.8899
        b3 = 0.7067
    
        light_yield = np.zeros(np.shape(proton_energy))
    
        light_yield[low_mask] = (
                                 a1 * proton_energy[low_mask]    + 
                                 b1 * proton_energy[low_mask]**2 + 
                                 c1 * proton_energy[low_mask]**3
                                 )
    
        light_yield[medium_mask] = (
                                    a2 + 
                                    b2 * proton_energy[medium_mask] +  
                                    c2 * proton_energy[medium_mask]**2 +
                                    d2 * proton_energy[medium_mask]**3
                                    )
    
        light_yield[high_mask] = (
                                  a3 + 
                                  b3 * proton_energy[high_mask]
                                  )
    elif function == 'stevanato':
        '''
        Light yield function from
        Stevanato, L., et al. 
        "Light output of EJ228 scintillation neutron detectors." 
        Applied Radiation and Isotopes 69.2 (2011): 369-372.
        '''
        
        L_0 = 0.606
        L_1 = 2.97
        light_yield = L_0*proton_energy**2/(proton_energy+L_1)
    if timer: elapsed_time(t_start)
    
    return s*light_yield


def get_kincut_function(tof, cut_factors=(1., 1., 1.), timer = False):
    '''
    Takes an array of times of flight [ns] and returns the corresponding 
    maximal/minimal light yield for each flight time (MeVee). The calculation
    is done by considering the TOFOR geometry to find the maximal/minimal
    scattering angles in the S1 detectors providing an upper and lower limit
    in the proton recoil energy in the S1. For the S2 there is only an upper
    upper limit, as there is no requirement in the scattering direction in the 
    S2s.

    Parameters
    ----------
    tof : ndarray,
        1D array of times-of-flight (ns)
    cut_factors : tuple of floats, optional
                Tuple of three factors (a, b, c) to apply to kinematic cuts. 
                Factors a and b are applied to lower and upper S1 kinematic
                cut, factor c is applied to upper S2 kinematic cut.
    timer : bool, optional
          If set to True, prints the time to execute the function.
                 
    Returns
    -------
    ly_S1_min : ndarray,
              Array of light yields (MeVee) corresponding to the lower 
              kinematic cut in S1.
    ly_S1_max :
              Array of light yields (MeVee) corresponding to the upper 
              kinematic cut in S1.
    ly_S2_max :
              Array of light yields (MeVee) corresponding to the upper 
              kinematic cut in S2.
            
    Examples
    --------
    >>> t_tof = [27, 28, 29] # (ns)
    >>> # Broaden S1 kinematic cuts by factor 0.7 down and 1.4 up
    >>> # Increase S2 cut by 20%
    >>> get_kincut_function(t_tof, (0.7, 1.4, 1.2)) 
    (array([0.41889004, 0.37579523, 0.33857325]),
     array([1.80448041, 1.62189197, 1.4616936 ]),
     array([4.91064379, 4.46936204, 4.07293829]))   
    '''
    
    if timer: t_start = elapsed_time()
    
    # Cast to numpy array
    tof = np.array(tof)
    
    # TOFOR geometry
    l_S2    = 0.35            # length of S2 [m]
    phi_max = np.deg2rad(115) # obtuse angle between S2 and line from S1 centre to S2 centre
    phi_min = np.deg2rad(65)  # acute  angle between S2 and line from S1 centre to S2 centre
    alpha   = np.deg2rad(30)  # angle between S1 and S2 centre w.r.t. centre of constant time-of-flight sphere
    r       = 0.7046          # time-of-flight sphere radius [m]
    
    # Calculate length between S1 and S2 centres [m]
    l = r * np.sin(np.pi - 2*alpha) / np.sin(alpha)

    # Maximum and minimum distances
    l_max = np.sqrt(l**2 + (l_S2/2)**2 - l*l_S2*np.cos(phi_max))
    l_min = np.sqrt(l**2 + (l_S2/2)**2 - l*l_S2*np.cos(phi_min))

    # Maximum and minimum scattering angles
    alpha_max = alpha + np.arccos((l**2 + l_min**2 - l_S2**2/4) / (2 * l * l_min))
    alpha_min = alpha - np.arccos((l**2 + l_max**2 - l_S2**2/4) / (2 * l * l_max))
    
    # Calculate cuts in proton recoil energy (MeV)
    J_to_MeV = 1E-6 / constant.electron_volt 
    E_S1_min = cut_factors[0]*0.5*constant.neutron_mass*(l_max/(tof*1E-9))**2*(1/np.cos(alpha_min)**2-1)*J_to_MeV
    E_S1_max = cut_factors[1]*0.5*constant.neutron_mass*(l_min/(tof*1E-9))**2*(1/np.cos(alpha_max)**2-1)*J_to_MeV
    E_S2_max = cut_factors[2]*0.5*constant.neutron_mass*(l_max/(tof*1E-9))**2*J_to_MeV

    # Translate to light yield
    ly_S1_max = light_yield_function(E_S1_max)
    ly_S1_min = light_yield_function(E_S1_min)
    ly_S2_max = light_yield_function(E_S2_max)

    if timer: elapsed_time(t_start, 'get_kincut_function()')
    return ly_S1_min, ly_S1_max, ly_S2_max

def kinematic_cuts(tof, energy_S1, energy_S2, cut_factors=(1., 1., 1.), timer = False):
    '''
    Performs kinematic cuts on the times of flight vs. energy for S1's and 
    S2's.

    Parameters
    ----------
    tof : ndarray,
        1D array of times-of-flight (ns)
    energy_S1 : ndarray, 
              1D array of S1 energies (MeVee)
    energy_S2 : ndarray, 
              1D array of S2 energies (MeVee)
    cut_factors : tuple of floats, optional
                Tuple of three factors (a, b, c) to apply to kinematic cuts. 
                Factors a and b are applied to lower and upper S1 kinematic
                cut, factor c is applied to upper S2 kinematic cut.
    timer : bool, optional
          If set to True, prints the time to execute the function.
                 
    Returns
    -------
    tof_cut : ndarray,       
            1D array of times of flight (ns) with kinematic cuts applied,
            len(tof_cut)<=len(tof).
    energy_S1_cut : ndarray,
                  1D array of energies (MeVee) for S1 with kinematic cuts 
                  applied, len(energy_S1_cut)<=len(energy_S1).
    energy_S2_cut : ndarray,
                  1D array of energies (MeVee) for S2 with kinematic cuts 
                  applied, len(energy_S2_cut)<=len(energy_S2).
            
    Examples
    --------
    >>> tof = [64, 65, 66, 67] # (ns)
    >>> energy_S1 = [0.05, 0.1, 0.2, 0.3] # (MeVee)
    >>> energy_S2 = [0.10, 0.2, 0.3, 0.4] # (MeVee)
    >>> kinematic_cuts(tof, energy_S1, energy_S2)
    array([64, 65]), 
    array([0.05, 0.1 ]), 
    array([0.1, 0.2])
    '''

    if timer: t_start = elapsed_time()
    # Cast to numpy arrays
    tof = np.array(tof) 
    energy_S1 = np.array(energy_S1)
    energy_S2 = np.array(energy_S2)
    
    # Run tof through get_kincut_function()
    S1_min, S1_max, S2_max = get_kincut_function(tof, cut_factors)

    # Compare measured energies with maximum/minimum energies for the given time of flight
    accept_inds = np.where((energy_S1 > S1_min) & (energy_S1 < S1_max) & (energy_S2 < S2_max))[0]

    if timer: elapsed_time(t_start, 'kinematic_cuts()')
    return tof[accept_inds], energy_S1[accept_inds], energy_S2[accept_inds]                 


def get_dictionaries(S = 0, fill = []):
    '''
    Returns dictionaries with S1/S2 detector names as keys.

    Parameters
    ----------
    S : str, optional
      String indicating which type of dictionary to return. Available options:
          
    fill : ndarray, optional,
         Sets the values of the dictionary to whatever is passed in "fill".
         
    Notes
    -----
    The input parameter S can be set to the following modes.\n
    S = 0: Returns two dictionaries, one for S1 and one for S2.\n
    S = 'S1': Returns a dictionary for S1.\n
    S = 'S2': Returns a dictionary for S2.\n
    S = 'merged': Returns single dictionary with S1/S2 detector names as keys.
    \n
    S = 'nested': Returns a nested dictionary where each S1 contains a 
    dictionary with S2 detector names as keys.
    
    Returns
    -------
    S1_dictionary : dict,       
                  Dictionary with S1 detector names as keys.
    S2_dictionary : dict,
                  Dictionary with S2 detector names as keys.
            
    Examples
    --------
    >>> s1, s2 = get_dictionaries(0, [1,2,3])
    >>> print(s1.keys())
    >>> print(s1.values())
    dict_keys(['S1_01', 'S1_02', 'S1_03', 'S1_04', 'S1_05'])
    dict_values([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    >>> print(s2.keys())
    dict_keys(['S2_01', 'S2_02', 'S2_03', 'S2_04', 'S2_05', 
               'S2_06', 'S2_07', 'S2_08', 'S2_09', 'S2_10', 
               'S2_11', 'S2_12', 'S2_13', 'S2_14', 'S2_15', 
               'S2_16', 'S2_17', 'S2_18', 'S2_19', 'S2_20', 
               'S2_21', 'S2_22', 'S2_23', 'S2_24', 'S2_25', 
               'S2_26', 'S2_27', 'S2_28', 'S2_29', 'S2_30', 
               'S2_31', 'S2_32'])
    '''

    S1_dictionary = {}
    for i in range(1, 6):
        dict_key = 'S1_0' + str(i)
        S1_dictionary.update({dict_key: fill.copy()})
    if S == 'S1': 
        return S1_dictionary
    
    S2_dictionary = {}
    for i in range(1, 33):
        if i < 10: dict_key = 'S2_0' + str(i)
        else: dict_key = 'S2_' + str(i)
        S2_dictionary.update({dict_key: fill.copy()})
    
    if S == 'S2': 
        return S2_dictionary
    if S == 'merged': 
        S1_dictionary.update(S2_dictionary)
        return S1_dictionary
    if S == 'nested': 
        return {'S1_01':get_dictionaries('S2', fill.copy()),
                'S1_02':get_dictionaries('S2', fill.copy()), 
                'S1_03':get_dictionaries('S2', fill.copy()), 
                'S1_04':get_dictionaries('S2', fill.copy()),
                'S1_05':get_dictionaries('S2', fill.copy())}
    if S == 'ADQ14':
        keys = list(get_dictionaries('merged').keys())
        return {key:fill.copy() for key in keys[:20]}
    if S == 'ADQ412':
        keys = list(get_dictionaries('merged').keys())
        return {key:fill.copy() for key in keys[20:]}
        
    return S1_dictionary, S2_dictionary

def get_boards():
    '''
    Returns an array of board names.
    '''
    return np.array(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'])

def get_channels():
    '''
    Returns an array of channel names.
    '''
    return np.array(['A', 'B', 'C', 'D'])

def find_ohmic_phase(shot_number, timer = False):
    '''
    Returns the JET time (s) at which the Ohmic phase is over for given shot number.

    Parameters
    ----------
    shot_number : int or string
                JET pulse number.
    timer : bool, optional
          If set to True, prints the time to execute the function.
                 
    Returns
    -------
    t_end : float,       
          JET time at which the Ohmic phase stops, i.e. the time at which any
          heating but Ohmic heatings is applied.
            
    Examples
    --------
    >>> find_ohmic_phase(98044)
    47.008
    '''

    if timer: t_start = elapsed_time()
    
    # Import NBI info
    nbi = ppf.ppfget(shot_number, dda = "NBI", dtyp = "PTOT")
    nbi_pow = nbi[2]
    nbi_tim = nbi[4]
    
    # Import ICRH info
    icrh = ppf.ppfget(shot_number, dda = "ICRH", dtyp = "PTOT")
    icrh_pow = icrh[2]
    icrh_tim = icrh[4]
    
    # Import LHCD info
    lhcd = ppf.ppfget(shot_number, dda = "LHCD", dtyp = "PTOT")
    lhcd_pow = lhcd[2]
    lhcd_tim = lhcd[4]
    
    # No NBI or ICRH or LHCD
    if (len(nbi_pow[nbi_pow > 0])   == 0 and 
        len(icrh_pow[icrh_pow > 0]) == 0 and 
        len(lhcd_pow[lhcd_pow > 0]) == 0): return 70.
    
    # Find where ICRH starts
    if len(icrh_pow[icrh_pow > 0]) > 0: icrh_start = icrh_tim[np.where(icrh_pow != 0)[0][0]]
    else: icrh_start = np.inf
    
    # Find where NBI starts
    if len(nbi_pow[nbi_pow > 0]) > 0: nbi_start = nbi_tim[np.where(nbi_pow != 0)[0][0]]
    else: nbi_start = np.inf
    
    # Find where LHCD starts
    if len(lhcd_pow[lhcd_pow > 0]) > 0: lhcd_start = lhcd_tim[np.where(lhcd_pow !=0)[0][0]]
    else: lhcd_start = np.inf

    first = np.argsort(np.array([nbi_start, icrh_start, lhcd_start]))[0]

    # Find which heating system started first
    if first == 0: t_end = nbi_start
    elif first == 1: t_end = icrh_start
    elif first == 2: t_end = lhcd_start
    
    if timer: elapsed_time(t_start, 'find_ohmic_phase()')
    return t_end



def elapsed_time(time_start = 0., timed_function = '', return_time = False):
    '''
    Optional timer for functions.

    Parameters
    ----------
    time_start : float, optional
               Starting time, used to calculate the elapsed time
    timed_function : str, optional
                   String used in print statement, typically set to the 
                   function being timed.
    return_time : bool, optional
                Returns the elapsed time
                                 
    Returns
    -------
    return_time : float,       
                Elapsed time (s)
            
    Examples
    --------
    >>> t_start = elapsed_time() # Start timer
    >>> time.sleep(1)
    >>> elapsed_time(t_start, 'example function') # Print elapsed time
    Elapsed time for example function: 1.00 sec.
    '''
    
    if not time_start: return time.time()
    else: print('Elapsed time for ' + timed_function + ': ' + '%.2f' %(time.time() - time_start) + ' sec.' )    
    if return_time: return time.time() - time_start

def background_subtraction(coincidences, tof_bins, energies_S1, S1_info, 
                           energies_S2, S2_info, disable_cuts, 
                           cut_factors=(1., 1., 1.), timer = False):
    '''
    Performs background subtraction of TOF spectrum using negative flight 
    times.
    
    Parameters
    ----------
    coincidences : ndarray,
                 1D array of times-of-flight (ns).
    tof_bins : ndarray,
             Bins used for TOF spectrum.
    energies_S1 : ndarray,
                1D array of energies (MeVee) for S1.
    S1_info : dict,
            Contains S1 information on energy bins, limits, etc.
    energies_S2 : ndarray,
                1D array of energies (MeVee) for S2.
    S2_info : dict,
            Contains S2 information on energy bins, limits, etc.
    disable_cuts : boolean,
                 If set to True, calculates the background as the average value
                 between TOF -100 to -50 ns. If set to False, takes kinematic
                 cuts into consideration.
    cut_factors : tuple of floats, optional
                Tuple of three factors (a, b, c) to apply to kinematic cuts. 
                Factors a and b are applied to lower and upper S1 kinematic
                cuts, factor c is applied to upper S2 kinematic cut.
    timer : bool, optional
          If set to True, prints the time to execute the function.
                                 
    Returns
    -------
    tof_bg : ndarray,
           1D array, background TOF component. Component is flipped to positive
           flight times. The entire component is returned (i.e. negative and 
           positive part) for the given input bins.
    '''
    
    if timer: t_start = elapsed_time()
    
    # Without kinematic cuts use average background between -100 ns and -50 ns
    tof_hist, _ = np.histogram(coincidences[(coincidences < -50) & (coincidences > -100)], tof_bins[(tof_bins < -50) & (tof_bins > -100)])
    mean_bg = np.mean(tof_hist)
    
    if disable_cuts: 
        tof_bg = np.ones(len(tof_bins)-1)
        tof_bg *= np.mean(tof_hist)    
        
    # Otherwise perform background averaging with kinematic cuts
    else: 
        # Select negative bins only
        tof_bins_n = tof_bins[tof_bins<0]
        
        # Calculate bin centres
        tof_bin_centres = tof_bins_n[1:]-np.diff(tof_bins_n)[0]/2
        S1_energy_bins = S1_info['energy bins']
        S1_energy_bin_width = np.diff(S1_energy_bins)[0]
        S1_energy_bin_centres = S1_energy_bins[1:]-S1_energy_bin_width/2
        
        S2_energy_bins = S2_info['energy bins']
        S2_energy_bin_width = np.diff(S2_energy_bins)[0]
        S2_energy_bin_centres = S2_energy_bins[1:]-S2_energy_bin_width/2
        
        # Histogram everything, only include tof < -20 ns (avoid muon peak)
        S1_hist, _, _ = np.histogram2d(coincidences, 
                                       energies_S1, 
                                       bins = [tof_bins_n[tof_bins_n<-20], 
                                               S1_info['energy bins']])
        S2_hist, _, _ = np.histogram2d(coincidences, 
                                       energies_S2, 
                                       bins = [tof_bins_n[tof_bins_n<-20], 
                                               S2_info['energy bins']])
        
        # Average along the x-axis
        S1_average = np.mean(S1_hist, axis = 0)
        S2_average = np.mean(S2_hist, axis = 0)

        S1_smooth = np.transpose(np.tile(S1_average, (len(tof_bin_centres), 1)))
        S2_smooth = np.transpose(np.tile(S2_average, (len(tof_bin_centres), 1)))
        
        
        # Get kinematic cuts for given TOF bin edges
        S1_min, S1_max, S2_max = get_kincut_function(tof_bin_centres, cut_factors)
        
        # S1
        s1_projection = np.zeros(len(tof_bin_centres))
        for i in range(0, len(tof_bin_centres)):
            # Select energy cut
            e_min = S1_min[i]
            e_max = S1_max[i]
            
            # Select the energy column corresponding to TOF centre
            col = S1_smooth[:, i]
            
            # Find upper/lower energy bin edge
            low = np.searchsorted(S1_energy_bins, e_min)  # lower cut
            high = np.searchsorted(S1_energy_bins, e_max) # upper cut
            
            # If we hit our maximum energy binning use max bin for projection
            if e_max >= S1_energy_bins[-1]: high -= 1
            if e_min >= S1_energy_bins[-1]: low -= 1
            lower = (low-1, low)   # (lower bin edge, upper bin edge) for lower cut
            upper = (high-1, high) # (lower bin edge, upper bin edge) for upper cut
            
            
            # If cuts are within a single bin
            if lower == upper:
                # Caclulate fraction of the bin to be added to projection
                fraction = (e_max-e_min)/S1_energy_bin_width
                to_project = fraction*col[low-1]
            # If cuts are at different bins
            else:
                # Calculate fraction of bins to be added
                fraction_low  = (S1_energy_bins[lower[1]]-e_min)/S1_energy_bin_width
                fraction_high = (e_max-S1_energy_bins[upper[0]])/S1_energy_bin_width
                to_project = fraction_low*col[low-1] + fraction_high*col[high-1]
            
            # Also add the bins which do not require fractions calculated
            to_project += col[lower[1]:upper[0]].sum()
            s1_projection[i] = to_project
        # S2
        s2_projection = np.zeros(len(tof_bin_centres))
        for i in range(0, len(tof_bin_centres)):
            # Select energy cut
            e_max = S2_max[i]
            
            # Select the energy column corresponding to TOF centre
            col = S2_smooth[:, i]
            
            # Find upper/lower energy bin edge
            high = np.searchsorted(S2_energy_bins, e_max) # upper cut
            
            # If we hit our maximum energy binning use max bin for projection
            if e_max >= S1_energy_bins[-1]: high -= 1
            upper = (high-1, high) # (lower bin edge, upper bin edge) for upper cut
            
            # Caclulate fraction of the bin to be added to projection
            fraction = (e_max-S2_energy_bins[upper[0]])/S2_energy_bin_width
            to_project = fraction*col[high-1]
            
            # Also add the bins which do not require fractions calculated
            to_project += col[:upper[0]].sum()
            s2_projection[i] = to_project
        
        sx_projection = s1_projection*s2_projection/mean_bg
        
        tof_bg = np.zeros(len(tof_bins)-1)
        tof_bg[0:len(sx_projection)] = sx_projection
        tof_bg[len(sx_projection)+1:] = np.flip(sx_projection)
    if timer: elapsed_time(t_start, 'background_subtraction()')
    

    return tof_bg

def set_plot_style():
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'nes_plots.mplstyle')
    matplotlib.rcParams['interactive'] = True
    plt.style.use(filename)



##################################
### Unfinished below this line ###
##################################

# Plot 1D histogram, allows looping several plots into same window with legend
def hist_1D_s(x_data, title = '', log = True, bins = 0, ax = -1, 
              normed = 0, density = False, x_label = 't$_{tof}$ [ns]', y_label = 'Counts', hist_type = 'standard', 
              alpha = 1, linewidth = 1, color = 'k', weights = None, linestyle = '-', timer = False):
    '''
    Example of how to use legend:
    fig = plt.figure('Time differences')
    ax = fig.add_subplot(111)
    bins = np.linspace(0, 1.15 * np.max(dt_ADQ14), 1000)
    hist_1D_s(dt_ADQ412, label = 'ADQ412', log = True, bins = bins, x_label = 'Time difference [ms]', ax = ax)
    hist_1D_s(dt_ADQ14,  label = 'ADQ14',  log = True, bins = bins, x_label = 'Time difference [ms]', ax = ax)
    '''
    if timer: t_start = elapsed_time()
    
    # Create bins if not given
    if bins is 0: bins = np.linspace(np.min(x_data), np.max(x_data), 100)
    

    
    bin_centres = bins[1:] - np.diff(bins) / 2
    hist = np.histogram(x_data, bins = bins, weights = weights)
    if normed: bin_vals = hist[0] / np.max(hist[0])
    else: bin_vals = hist[0]
    
    # Plot with uncertainties
    if hist_type == 'standard':
        cap_size = 1.5
        line_width = 1
        marker = '.'
        marker_size = 1.5
        plt.plot(bin_centres, 
                 bin_vals, 
                 marker = marker, 
                 alpha = alpha,
                 markersize = marker_size,
                 color = color,
                 linestyle = 'None')
        plt.errorbar(bin_centres, 
                     bin_vals, 
                     np.sqrt(bin_vals), 
                     color = color, 
                     alpha = alpha,
                     elinewidth = line_width,
                     capsize = cap_size,
                     linestyle = 'None')
        plt.yscale('log')
        ax = -1
    else:
        plt.hist(bin_centres, bins = bins, weights = bin_vals, log = log,
                 histtype = hist_type, alpha = alpha, linewidth = linewidth,
                 color = color, linestyle = linestyle, density = density)        

    plt.title(title)
    plt.xlim([bins[0], bins[-1]])
    
    plt.xlabel(x_label, fontsize = 14)
    plt.ylabel(y_label, fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)

    
    # Include legend
    if ax != -1:
        
        handles, labels = ax.get_legend_handles_labels()
        new_handles = [Line2D([], [], c = color) for h in handles]

        plt.legend(handles=new_handles, labels=labels, loc = 'upper right')
    
    if timer: elapsed_time(t_start, 'hist_1D_s()')
    return hist
    


def plot_2D(times_of_flight, energy_S1, energy_S2, bins_tof = np.arange(-199.8, 200, 0.4), 
            S1_info = None, S2_info = None, tof_lim = np.array([-150, 200]), title = '', tof_bg_component = 0,
            log = True, interactive_plot = False, projection = 0, disable_cuts = False,
            times_of_flight_cut = 0, energy_S1_cut = 0, energy_S2_cut = 0, disable_bgs = False, 
            weights = False, hist2D_S1 = None, hist2D_S2 = None, sum_shots = False, 
            proton_recoil = False, pulse_height_spectrum = False, integrated_charge_spectrum = False,
            cut_factors = (1., 1., 1.), timer = False):
    ''' This is a mess, I'm sorry (it does work though).
    Plots 2D histogram of TOF vs energy with projections onto time and energy axis.
    times_of_flight: 1D array of times of flight.
    energy_S1: 1D array of energies for S1
    energy_S2: 1D array of energies for S2
    bins_tof: bins for 1D time of flight spectrum
    bins_energy: bins for 1D energy spectrum
    bins_2D: bins for 2D spectrum of energy vs. time of flight
    energy_lim: set energy plotting limits
    title: title
    log: set log scale
    interactive_plot: set to true to set cuts in each spectrum
    projection: used in replot_projections() function, allows for red lines to be plotted along the limits of the cuts
    '''
    if timer: t_start = elapsed_time()
    
    # Configure plots
    cap_size = 1.5
    line_width = 1
    marker = '.'
    marker_size = 1
    
    # Add lines for cuts
    if projection != 0: add_lines = True
    else: add_lines = False
    
    fig = plt.figure(title, figsize=(7,5))
    
    '''
    TOF projection
    '''
    # If kinematic cuts are applied
    if not disable_cuts: 
        tof = times_of_flight_cut
        # Only plot projection of energies
        erg_S1 = energy_S1_cut
        erg_S2 = energy_S2_cut
    else: 
        tof = times_of_flight
        erg_S1 = energy_S1
        erg_S2 = energy_S2
    
    # If light yield function is enabled, plot proton recoil energy instead of light yield
    if proton_recoil:
        erg_S1 = inverted_light_yield(erg_S1)
        erg_S2 = inverted_light_yield(erg_S2)
        erg_unit = 'MeV'
    elif pulse_height_spectrum:
        erg_unit = 'a.u.'
    elif integrated_charge_spectrum:
        erg_unit = 'a.u.'
    else: erg_unit = '$MeV_{ee}$'
        
    TOF_fig = plt.subplot(326)
    bins_tof_centres = bins_tof[1:] - np.diff(bins_tof)[0] / 2
    if weights: 
        TOF_hist = tof
        TOF_plot = tof
        
    else: 
        TOF_hist, _ = np.histogram(tof, bins = bins_tof)
        TOF_plot = TOF_hist
        
    # Apply background subtraction
    if not disable_bgs:
        # Remove background from binned values
        TOF_plot = TOF_hist - tof_bg_component
        
        # Plot background component + fit
        plt.plot(bins_tof_centres, tof_bg_component, 'r--')     
        plt.plot(bins_tof_centres[bins_tof_centres < 0], 
                 TOF_hist[bins_tof_centres < 0], 
                 marker = marker,
                 markersize = marker_size,
                 color = 'r',
                 linestyle = 'None')
        plt.errorbar(bins_tof_centres[bins_tof_centres < 0],
                     TOF_hist[bins_tof_centres < 0],
                     yerr = np.sqrt(TOF_hist[bins_tof_centres < 0]),
                     linestyle = 'None',
                     capsize = cap_size,
                     elinewidth = line_width,
                     color = 'r')
        
    # Plot        
    plt.plot(bins_tof_centres, 
             TOF_plot,
             marker = marker,
             markersize = marker_size,
             color = 'k',
             linestyle = 'None')
        
    plt.errorbar(bins_tof_centres, 
                 TOF_plot,
                 yerr = np.sqrt(TOF_hist), 
                 linestyle = 'None',
                 capsize = cap_size,
                 elinewidth = line_width,
                 color = 'k')
    plt.yscale('log')
    
    # Get current axis
    ax_TOF = plt.gca() 
    ax_TOF.set_xlabel('$t_{TOF}$ (ns)')
    ax_TOF.set_ylabel('Counts')
    tof_x_low = 20
    tof_x_high = 80
    ax_TOF.set_xlim(tof_x_low, tof_x_high)
    ax_TOF.set_ylim(bottom = np.min(TOF_hist) / 2 + 1)
    
    # Add lines for interactive plot
    if add_lines:
        print('adding lines')
        big_value = 100000000
        if projection['proj'] == 'times-of-flight': 
            proj_lims = projection['limits']
            plt.plot([proj_lims[0], proj_lims[0]], [-big_value, big_value], '--r')
            plt.plot([proj_lims[1], proj_lims[1]], [-big_value, big_value], '--r')
    

    '''
    S1 2D spectrum
    '''
    plt.subplot(322, sharex = ax_TOF)
    
    # Set white background
    my_cmap = plt.cm.jet
    my_cmap.set_under('w', 1)
    
    # Get 2D binning
    bins_2D_S1 = [bins_tof, S1_info['energy bins']]
    bins_2D_S2 = [bins_tof, S2_info['energy bins']]
    
    # Find the max value for the z-axis on the 2D plot (no kinematic cuts applied)
    if weights:
        S1_max = np.max(hist2D_S1)
        S2_max = np.max(hist2D_S2)
    else:
        S1_max = np.max(plt.hist2d(times_of_flight, energy_S1, bins=bins_2D_S1, cmap=my_cmap, vmin=1)[0])
        S2_max = np.max(plt.hist2d(times_of_flight, energy_S2, bins=bins_2D_S2, cmap=my_cmap, vmin=1)[0])
    if S1_max >= S2_max: vmax = S1_max
    else: vmax = S2_max
    
    # Plot first 2D histogram (no kinematic cuts applied)
    bins_energy_centres_S1 = S1_info['energy bins'][1:] - np.diff(S1_info['energy bins'])[0] / 2
    bins_energy_centres_S2 = S2_info['energy bins'][1:] - np.diff(S2_info['energy bins'])[0] / 2

    if weights:        
        # Create data set to fill 2D spectrum with one count in each bin
        tof_repeated = np.tile(bins_tof_centres, len(bins_energy_centres_S1))
        energy_repeated = np.repeat(bins_energy_centres_S1, len(bins_tof_centres))
        weights2D_S1 = np.ndarray.flatten(np.transpose(hist2D_S1))
        # Create 2D histogram using weights
        hist2d_S1 = plt.hist2d(tof_repeated, 
                               energy_repeated, 
                               bins = bins_2D_S1, 
                               weights = weights2D_S1,
                               cmap = my_cmap, 
                               vmin = 1, 
                               vmax = vmax)[0]
    else:
        hist2d_S1 = plt.hist2d(times_of_flight, 
                               energy_S1, 
                               bins = bins_2D_S1, 
                               cmap = my_cmap, 
                               vmin = 1, 
                               vmax = vmax)[0]
    ax_S1_2D = plt.gca()
    plt.setp(ax_S1_2D.get_xticklabels(), visible = False)
    plt.setp(ax_S1_2D.get_yticklabels(), visible = False)
    
    # Add lines for interactive plot
    if add_lines:
        if projection['proj'] == 'time-of-flight and S1 energy':
            proj_lims_tof = projection['limits'][0]
            proj_lims_E = projection['limits'][1]
            plt.plot([proj_lims_tof[0], proj_lims_tof[0]], [-big_value, big_value], '--r')
            plt.plot([proj_lims_tof[1], proj_lims_tof[1]], [-big_value, big_value], '--r')
            plt.plot([-big_value, big_value], [proj_lims_E[0], proj_lims_E[0]], '--r')
            plt.plot([-big_value, big_value], [proj_lims_E[1], proj_lims_E[1]], '--r')
        
    # Add lines for kinematic cuts
    if not disable_cuts:
        tof_axis_p = np.linspace(0.1, 500, 500)
        tof_axis_n = np.linspace(-0.1, -500, 500)
        S1_min, S1_max, S2_max = get_kincut_function(tof_axis_p, cut_factors)
        plt.plot(tof_axis_p, S1_min, 'r-')
        plt.plot(tof_axis_p, S1_max, 'r-')
        plt.plot(tof_axis_n, S1_min, 'r-')
        plt.plot(tof_axis_n, S1_max, 'r-')
        
    '''
    S2 2D spectrum
    '''
    # Plot second 2D histogram (no kinematic cuts applied)
    plt.subplot(324, sharex = ax_TOF)
    if weights:        
        # Create 2D histogram using weights
        weights2D_S2 = np.ndarray.flatten(np.transpose(hist2D_S2))
        hist2d_S2 = plt.hist2d(tof_repeated, 
                               energy_repeated, 
                               bins = bins_2D_S2, 
                               weights = weights2D_S2,
                               cmap = my_cmap, 
                               vmin = 1, 
                               vmax = vmax,
                               norm = matplotlib.colors.LogNorm())[0]
    else:
        hist2d_S2 = plt.hist2d(times_of_flight, 
                               energy_S2, 
                               bins = bins_2D_S2, 
                               cmap = my_cmap, 
                               vmin = 1, 
                               vmax = vmax,
                               norm = matplotlib.colors.LogNorm())[0]
    ax_S2_2D = plt.gca()
    plt.setp(ax_S2_2D.get_xticklabels(), visible = False)
    plt.setp(ax_S2_2D.get_yticklabels(), visible = False)
    ax_S2_2D.set_xlim([tof_x_low, tof_x_high])

    if add_lines:
        if projection['proj'] == 'time-of-flight and S2 energy':
            proj_lims_tof = projection['limits'][0]
            proj_lims_E = projection['limits'][1]
            plt.plot([proj_lims_tof[0], proj_lims_tof[0]], [-big_value, big_value], '--r')
            plt.plot([proj_lims_tof[1], proj_lims_tof[1]], [-big_value, big_value], '--r')
            plt.plot([-big_value, big_value], [proj_lims_E[0], proj_lims_E[0]], '--r')
            plt.plot([-big_value, big_value], [proj_lims_E[1], proj_lims_E[1]], '--r')
    
    # Add lines for kinematic cuts
    if not disable_cuts: 
        plt.plot(tof_axis_p, S2_max, 'r-')
        plt.plot(tof_axis_n, S2_max, 'r-')
    '''
    Colour bar
    '''
    ax_colorbar = fig.add_axes([0.18, 0.26, 0.28, 0.03])

    plt.colorbar(ax = ax_S1_2D, cax = ax_colorbar, orientation = 'horizontal')
    

    '''
    S2 energy projection
    '''
    plt.subplot(323, sharey = ax_S2_2D)
    if weights: 
        
        plt.plot(erg_S2, 
                 bins_energy_centres_S2, 
                 marker = marker,
                 markersize = marker_size,
                 color = 'k',
                 linestyle = 'None')
        plt.errorbar(erg_S2, 
                     bins_energy_centres_S2, 
                     xerr = np.sqrt(erg_S2), 
                     linestyle = 'None',
                     capsize = cap_size,
                     elinewidth = line_width,
                     color = 'k')
        S2_E_hist = erg_S2
    else: 
        S2_E_hist, _ = np.histogram(erg_S2, bins = S2_info['energy bins'])
        plt.plot(S2_E_hist,
                 bins_energy_centres_S2,
                 marker = marker,
                 markersize = marker_size,
                 color = 'k',
                 linestyle = 'None')
        plt.errorbar(S2_E_hist, 
                     bins_energy_centres_S2, 
                     xerr = np.sqrt(S2_E_hist),
                     linestyle = 'None',
                     capsize = cap_size,
                     elinewidth = line_width,
                     color = 'k')
        
    ax_S2_E = plt.gca()
    ax_S2_E.set_xlabel('Counts')
    ax_S2_E.set_ylim(S2_info['energy limits'])
    ax_S2_E.set_xscale('log')

    if add_lines:
        if projection['proj'] == 'S2':
            proj_lims = projection['limits']
            plt.plot([-big_value, big_value], [proj_lims[0], proj_lims[0]], '--r')
            plt.plot([-big_value, big_value], [proj_lims[1], proj_lims[1]], '--r')
    ax_S2_E.set_ylim(S2_info['energy limits'])
    '''
    S1 energy projection
    '''
    plt.subplot(321, sharey = ax_S1_2D, sharex = ax_S2_E)
    
    if add_lines:
        if projection['proj'] == 'S1':
            proj_lims = projection['limits']
            plt.plot([-big_value, big_value], [proj_lims[0], proj_lims[0]], '--r')
            plt.plot([-big_value, big_value], [proj_lims[1], proj_lims[1]], '--r')
    
    if weights:
        plt.plot(erg_S1, 
                 bins_energy_centres_S1, 
                 marker = marker,
                 markersize = marker_size,
                 color = 'k',
                 linestyle = 'None')
        plt.errorbar(erg_S1, 
                     bins_energy_centres_S1, 
                     xerr = np.sqrt(erg_S1), 
                     linestyle = 'None',
                     capsize = cap_size,
                     elinewidth = line_width,
                     color = 'k')
        S1_E_hist = erg_S1
        
    else: 
        S1_E_hist, _ = np.histogram(erg_S1, bins = S1_info['energy bins'])
        plt.plot(S1_E_hist, 
                 bins_energy_centres_S1, 
                 marker = marker,
                 markersize = marker_size,
                 color = 'k',
                 linestyle = 'None')
        plt.errorbar(S1_E_hist, 
                     bins_energy_centres_S1, 
                     xerr = np.sqrt(S1_E_hist),
                     linestyle = 'None',
                     capsize = cap_size,
                     elinewidth = line_width,
                     color = 'k')
    ax_S1_E = plt.gca()
    plt.setp(ax_S1_E.get_xticklabels(), visible = False)
    ax_S1_E.set_ylim(S1_info['energy limits'])
    ax_S1_E.set_xscale('log')
    
    # Set the x-axis limits
    x_lower = 0.1
    S2_events = S2_E_hist
    S1_events = S1_E_hist
    if np.sum(S2_events) == 0: x_upper = 1
    elif np.max(S2_events) >= np.max(S1_events): x_upper = np.max(S2_events)
    else: x_upper = np.max(S1_events)
    
    if np.sum(S2_events) == 0 or np.sum(S1_events) == 0: x_lower  = 0
    elif np.min(S2_events[S2_events > 0] <= np.min(S1_events[S1_events > 0])): 
        x_lower = np.min(S2_events[S2_events > 0])
    else: x_lower = np.min(S1_events[S1_events > 0])
    ax_S2_E.set_xlim([x_lower, x_upper])
    
    # Set x,y-label
    fig.text(0.12, 0.68, f'Deposited energy ({erg_unit})', va='center', rotation='vertical')
    fig.text(0.3, 0.94, title, va = 'center', ha = 'center')
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    
    '''
    Begin interactive plotting
    '''
    plt.show(block = False)
    if interactive_plot:
        plt.show(block = False)
        while True:
            print('\nSelect one of the following panels and set upper and lower limits to project the selected limit onto the other dimensions.')
            print('TL - top left')
            print('TR - top right')
            print('ML - middle left')
            print('MR - middle right')
            print('BR - bottom right')
            print('Type exit to exit.')
            panel_choice = input('Select panel (TL, TR, ML, MR, BR): ')
            if panel_choice in ['exit', 'Exit', 'EXIT']: break
            
            # Cut in 2D Spectrum
            if panel_choice in ['TR', 'tr', 'MR', 'mr']:
                tof_choice = input('Type limits for time axis ("lower upper"): ')
                energy_choice = input('Type limits for energy axis ("lower upper"): ')
                
                # Find space in user input
                find_space_tof = tof_choice.find(' ')
                find_space_energy = energy_choice.find(' ')
                
                # Transform to array of two floats [float_1, float_2]
                tof_choice = [float(tof_choice[0:find_space_tof]), float(tof_choice[find_space_tof + 1:])]
                energy_choice = [float(energy_choice[0:find_space_energy]), float(energy_choice[find_space_energy + 1:])]
                limits = [tof_choice, energy_choice]    
            
            # Cut in 1D spectrum
            elif panel_choice in ['TL', 'tl', 'ML', 'ml', 'BR', 'br']:
                if panel_choice in ['BR', 'br']: limits = input('Type limits for time axis ("lower upper"): ')
                elif panel_choice in ['TL', 'tl']: limits = input('Type limits for S1 energy axis ("lower upper"): ')
                else: limits = input('Type limits for S2 energy axis ("upper lower"): ')    
                
                # Find space in user input
                find_space  = limits.find(' ')
                limits = [float(limits[0:find_space]), float(limits[find_space + 1:])]
                
            else: 
                print('Invalid choice.')
                continue
            # Replot with new projections
            replot_projections(limits = limits, panel_choice = panel_choice, 
                               times_of_flight = tof, energy_S1 = erg_S1, 
                               energy_S2 = erg_S2, bins_tof = bins_tof, 
                               S1_info = S1_info, S2_info = S2_info, log = log,
                               disable_cuts = disable_cuts, disable_bgs = True, 
                               energy_S1_cut = erg_S1, energy_S2_cut = erg_S2, 
                               times_of_flight_cut = tof, 
                               proton_recoil = proton_recoil,
                               pulse_height_spectrum = pulse_height_spectrum,
                               integrated_charge_spectrum = integrated_charge_spectrum)
    elif sum_shots: plt.close(fig)

    if timer: elapsed_time(t_start, 'plot_2D()')
    return TOF_hist, S1_E_hist, S2_E_hist, hist2d_S1, hist2d_S2
    
def replot_projections(limits, panel_choice, times_of_flight, energy_S1, 
                       energy_S2, bins_tof, S1_info = None, S2_info = None,
                       log = True, disable_cuts = False, disable_bgs = False, 
                       energy_S1_cut = 0, energy_S2_cut = 0, 
                       times_of_flight_cut = 0, proton_recoil = False, 
                       pulse_height_spectrum = False,
                       integrated_charge_spectrum = False, 
                       cut_factors = (1., 1., 1.)):
    '''
    Replot the spectra with a cut on one of the energy projections
    limits: limits of cuts for projections. 1x2 array for 1D spectrum, 2x2 array for 2D spectrum
    panel_choice: panel to be cut (BR - bottom right, ML - middle left, TR - top right etc.)
    times_of_flight: 1D array of times of flight
    energy_S1: 1D array of energies for S1
    energy_S2: 1D array of energies for S2
    '''
    
    # If cut has been made in one of the 1D spectra
    if np.shape(limits) == (2, ):
        # Make cut in S1 or S2 energy
        if panel_choice in ['TL', 'tl']: 
            if not disable_cuts: eoi = energy_S1_cut
            else: eoi = energy_S1
            det = 'S1'
            uni = 'E'
            proj = {'S1_energy':[]}
        if panel_choice in ['ML', 'ml']: 
            if not disable_cuts: eoi = energy_S2_cut
            else: eoi = energy_S2
            det = 'S2'
            uni = 'E'

        if panel_choice in ['BR', 'br']:
            if not disable_cuts: eoi = times_of_flight_cut
            else: eoi = times_of_flight
            det = 'times-of-flight'
            uni = 'tof'
        
        # Find all events within this cut
        inds = np.where((eoi >= limits[0]) & (eoi <= limits[1]))[0]
        
        title = f'Cut in {det}\n{round(limits[0], 2)} < {uni} < {round(limits[1], 2)}'
        
    # If the cut has been made in one of the 2D spectra
    else:
        if panel_choice in ['TR', 'tr']:
            eoi = energy_S1
            det = 'time-of-flight and S1 energy'
        elif panel_choice in ['MR', 'mr']:
            eoi = energy_S2
            det = 'time-of-flight and S2 energy'
        
        # Select events which fulfill the criteria in tof and energy
        tof_choice = limits[0]
        energy_choice = limits[1]
        
        inds = np.where((times_of_flight >= tof_choice[0]) & (times_of_flight <= tof_choice[1]) &
                        (eoi >= energy_choice[0]) & (eoi <= energy_choice[1]))
        title = f'Cut in {det}\n{round(tof_choice[0], 2)} < tof < {round(tof_choice[1], 2)}, {round(energy_choice[0], 2)} < E < {round(energy_choice[1], 2)}'

       
    proj = {'proj':det, 'limits':limits}
    
    # Replot using the cut
    times_of_flight = times_of_flight[inds]
    energy_S1 = energy_S1[inds]
    energy_S2 = energy_S2[inds]
    
    plot_2D(times_of_flight = times_of_flight, energy_S1 = energy_S1, 
              energy_S2 = energy_S2, bins_tof = bins_tof, S1_info = S1_info, 
              S2_info = S2_info, times_of_flight_cut = times_of_flight, 
              energy_S1_cut = energy_S1, energy_S2_cut = energy_S2_cut,
              interactive_plot = False, disable_cuts = disable_cuts, 
              disable_bgs = disable_bgs, title = title, projection = proj, 
              log = log, proton_recoil = proton_recoil, 
              pulse_height_spectrum = pulse_height_spectrum,
              integrated_charge_spectrum = integrated_charge_spectrum, 
              cut_factors = cut_factors)

    
def get_energy_calibration_(areas, detector_name, timer = False):
    '''
    Return the deposited energy (MeVee) in the given detector using the energy
    calibration given in the energy calibration folder. 
    
    Notes
    -----
    Used temporarily for the new energy calibration where we use smaller
    integration limits (10-30 ns) when calculating the pulse waveform areas.
    
    Parameters
    ----------
    areas : ndarray,
          1D array of pulse areas. 
    detector_name : string,
                  Detector name corresponding to the pulse areas being parsed.
    timer : bool, optional
          If set to True, prints the time to execute the function.                 
                    
    Returns
    -------
    energy_array : ndarray,
                 1D array of deposited energies in MeVee.
            
    Examples
    --------
    >>> energies = get_energy_calibration(areas, 'S1_01')
    array([0.00134, 0.23145, ..., 0.02134])
    ''' 
    
    if timer: t_start = elapsed_time()
    
    raise_exception = False
    # Load calibration data for given detector
    if detector_name[0:2] == 'S1':
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../energy_calibration/energy_calibration_S1_.txt')
        cal = np.loadtxt(filename, usecols = (0,1))[int(detector_name[3:]) - 1]

        cal_factor = 3000.
    elif detector_name[0:2] == 'S2':
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, '../energy_calibration/energy_calibration_S2_.txt')
        cal = np.loadtxt(filename, usecols = (0,1))[int(detector_name[3:]) - 1]
        
        if int(detector_name[3:]) <= 15: cal_factor = 3000.
        elif int(detector_name[3:]) > 15: cal_factor = 350.
        else: raise_exception = True   
    else: raise_exception = True
    if raise_exception: raise Exception('Please supply the detector type as the second parameter (SX = \'S1_x\' x = [01, 05] or SX = \'S2_x\' x = [01, 32])')        
    
    # Calculate energy from area
    energy_array = (cal[0] + cal[1] * areas / cal_factor ) / 1000.
    if timer: elapsed_time(t_start, 'get_energy_calibration()')
    return energy_array
    
def print_help():
    print('\nPlease supply the shot number.')
    print('Example: python create_TOF.py --JPN 94217')
    print('Set the shot number to 0 to run the latest shot.')
    print('\nAdditional optional arguments:')
    print('--input-file my_input.txt: Input arguments from my_input.txt are used.')
    print('--1D-spectrum: Only plot time-of-flight spectrum.')
    print('--remove-doubles mode:\n  \
mode = 0: Remove all events which have produced a coincidence between two S1\'s.\n  \
mode = 1: Only plot events which have produced a coincidence between two S1\'s')
    print('--save-data: Save the data as a python pickle with file name \"(shot_number)_(t0)_(t1).pickle\".')
    print('--save-NES: Save histogram data as a python pickle with file name \"(shot_number)_(t0)_(t1).pickle\".')
    print('--time-range start stop: Only plot the data between \"start\" and \"stop\" seconds into the shot. \"start\" and \"stop\" are given in number of seconds since PRE.')
    print('--time-range-file: Selects shots and time ranges from separate input file (see input_files/time_ranges.txt for example. Set --disable-plots and --disable-scratch to run without user interaction.')
    print('--disable-cuts: Plot the data without any kinematic cuts.')
    print('--apply-cut-factors a b c: Apply factors to kinematic cuts. Factors a and b are applied to lower and upper S1 cuts, factor c is applied to S2 upper cut.')
    print('--disable-bgs: Plot the data without background subtracting the time-of-flight spectrum.')
    print('--disable-plots: Disables plotting.')
    print('--disable-detectors: Analysis is not performed on detectors specified by user. Example: --disable-detectors S1_01 S1_02 S2_01')
    print('--disable-boards: Analysis is not performed on boards specified by user. Example: --disable-boards 7 8 9')
    print('--disable-cleanup: Disables cleanup() function responsible for removing bad pulses.')
    print('--enable-detectors: Analysis is only performed on detectors speecified by user. Example: --enable-detectors S1_01 S2_01')
    print('--ohmic-spectrum: Analysis is only performed for the Ohmic phase of the shot.')
    print('--run-timer: Print the elapsed time for each function.')
    print('--software-thresholds thresholds.txt: Set software energy thresholds (in MeVee) specified in thresholds.txt (in MeV). Example: --software-thresholds input_files/thresholds.txt')
    print('--proton-recoil-energy: Convert the energy axis from MeVee to MeV scale.')
    print('--pulse-height-spectrum: Use maxima of pulses for the energy axis.')
    print('--integrated-charge-spectrum: Use integrated charge for the energy axis.')
    print('--help: Print this help text.')
    
if __name__=='__main__':
    pass







