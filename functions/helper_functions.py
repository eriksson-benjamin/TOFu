#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:13:12 2024

@author: beriksso
"""

import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import os
import tofu_functions as dfs


def plot_nes_pickle(f_name):
    """
    Generate a plot from a pickle file created using the --save-NES argument.

    Parameters
    ----------
    f_name : str
        The file name of the pickle file containing the histogram data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    (ax1, ax2, ax3) : tuple of matplotlib.axes.Axes
        Tuple containing the three axes used in the subplot.
    """    
    set_nes_plot_style()
    dat = import_data(f_name)

    # Plot 2d histogram
    fig, (ax1, ax2, ax3) = nes_pickle_plotter(*dat)

    return fig, (ax1, ax2, ax3)


def nes_pickle_plotter(t_counts, t_bins, t_bgr, e_bins_S1, e_bins_S2, matrix_S1,
                       matrix_S2, inp_args):
    """
    Plot 2D histograms for S1, S2, and TOF projection.

    This function creates a plot with three subplots: one for S1, one for S2,
    and a TOF projection plot. It also applies kinematic cuts if specified.

    Parameters
    ----------
    t_counts : array_like
        TOF (time of flight) counts.
    t_bins : array_like
        TOF bins.
    t_bgr : array_like
        Background level for TOF projection.
    e_bins_S1 : array_like
        Energy bins for S1.
    e_bins_S2 : array_like
        Energy bins for S2.
    matrix_S1 : array_like
        2D histogram matrix for S1.
    matrix_S2 : array_like
        2D histogram matrix for S2.
    inp_args : list
        Input arguments that control plot features such as kinematic cuts.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object.
    (ax1, ax2, ax3) : tuple of matplotlib.axes.Axes
        Tuple containing the three axes used in the subplot for S1, S2, and TOF.
    """    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    fig.set_size_inches(4, 12)

    # Set colorbar min/max
    vmin = 1
    vmax = (matrix_S1.max() if matrix_S1.max() > matrix_S2.max() else
            matrix_S2.max())
    normed = matplotlib.colors.LogNorm(vmin, vmax)

    # Set white background
    try:
        my_cmap = matplotlib.cm.get_cmap('jet').copy()
    except:
        my_cmap = plt.cm.jet
    my_cmap.set_under('w', 1)

    # Plot S1 2D histogram
    # --------------------
    x_r, y_r, weights = setup_matrix(matrix_S1, t_bins, e_bins_S1)
    ax1.hist2d(x_r, y_r, bins=(t_bins, e_bins_S1), weights=weights,
               cmap=my_cmap, norm=normed)

    # Plot S2 2D histogram
    # --------------------
    x_r, y_r, weights = setup_matrix(matrix_S2, t_bins, e_bins_S2, )
    ax2.hist2d(x_r, y_r, bins=(t_bins, e_bins_S1), weights=weights,
               cmap=my_cmap, norm=normed)

    # Plot TOF projection
    # -------------------
    ax3.plot(t_bins, t_counts, 'k.', markersize=1)
    ax3.errorbar(t_bins, t_counts, np.sqrt(t_counts), color='k',
                 linestyle='None')

    # Plot background component
    ax3.plot(t_bins, t_bgr, 'C2--')

    # Add lines for kinematic cuts
    if '--disable-cuts' not in inp_args:
        S1_min, S1_max, S2_max = get_kinematic_cuts(inp_args, t_bins)
        ax1.plot(t_bins, np.array([S1_min, S1_max]).T, 'r')
        ax2.plot(t_bins, S2_max, 'r')

    # Configure plot
    # --------------
    ax1.set_ylabel('$E_{ee}^{S1}$ $(MeV_{ee})$')
    ax2.set_ylabel('$E_{ee}^{S2}$ $(MeV_{ee})$')
    ax3.set_ylabel('counts')
    ax3.set_xlabel('$t_{TOF}$ (ns)')

    y1 = (0, 2.3)
    y2 = (0, 6)
    x3 = (-100, 100)
    # y3 = (10, 2E3)
    ax1.set_ylim(y1)
    ax2.set_ylim(y2)
    ax3.set_yscale('log')
    # ax3.set_yticks([1, 1E1, 1E2, 1E3, 1E4, 1E5, 1E6, 1E7])
    # ax3.set_xlim(x3)
    ax3.set_ylim(bottom=1)

    bbox = dict(facecolor='white', edgecolor='black')
    ax1.text(0.89, 0.88, '(a)', transform=ax1.transAxes, bbox=bbox)
    ax2.text(0.89, 0.88, '(b)', transform=ax2.transAxes, bbox=bbox)
    ax3.text(0.89, 0.88, '(c)', transform=ax3.transAxes, bbox=bbox)

    # Add colorbar
    # ------------
    fig.subplots_adjust(top=0.8)
    cbar_ax = fig.add_axes([0.2, 0.84, 0.73, 0.02])
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=normed)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    try:
        fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    except:
        print('Colorbar feature not available on the JDC.')

    plt.subplots_adjust(hspace=0.1)

    return fig, (ax1, ax2, ax3)



def import_data(file_name):
    """
    Import and process 2D histogram data from a given pickle file.

    Extracts various components from a pickle file, including TOF counts,
    background levels, S1 and S2 2D histograms, and binning information.

    Parameters
    ----------
    file_name : str
        The path to the pickle file.

    Returns
    -------
    t_counts : array_like
        TOF (time of flight) counts.
    t_bins : array_like
        TOF bins.
    t_bgr : array_like
        Background level for TOF projection.
    e_S1 : array_like
        Energy bin centers for S1.
    e_S2 : array_like
        Energy bin centers for S2.
    m_S1 : array_like
        2D histogram matrix for S1.
    m_S2 : array_like
        2D histogram matrix for S2.
    input_args : list
        Input arguments extracted from the pickle file.
    """
    p = unpickle(file_name)
    input_args = p['input_arguments']

    # Projection on t_tof axis
    t_counts = p['counts']

    # Reshape background
    bgr = p['bgr_level']
    t_bgr = np.append(np.flip(bgr[1:]), bgr)

    # 2D matrices
    m_S1 = p['hist2d_S1']
    m_S2 = p['hist2d_S2']

    # Bins
    t_bins = p['bins']
    e_bins_S1 = p['S1_info']['energy bins']
    e_bins_S2 = p['S2_info']['energy bins']

    # Calculate bin centres
    e_S1 = e_bins_S1[1:] - np.diff(e_bins_S1) / 2
    e_S2 = e_bins_S2[1:] - np.diff(e_bins_S2) / 2

    return t_counts, t_bins, t_bgr, e_S1, e_S2, m_S1, m_S2, input_args


def unpickle(file_name):
    """
    Unpickle the given file and return the content.

    Parameters
    ----------
    file_name : str
        The path to the pickle file.

    Returns
    -------
    A : object
        The unpickled content.
    """    
    with open(file_name, 'rb') as handle:
        A = pickle.load(handle)
        return A


def set_nes_plot_style():
    """
    Set the plot style. This function applies the plot style defined in the
    'nes_plots.mplstyle' file.
    """    
    matplotlib.rcParams['interactive'] = True
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'nes_plots.mplstyle')
    plt.style.use(filename)
    
    
def setup_matrix(matrix, x_bins, y_bins):
    """
    Flatten and repeat bin arrays to prepare data for 2D histograms.

    Parameters
    ----------
    matrix : array_like
        The 2D histogram matrix to process.
    x_bins : array_like
        The bin edges for the x-axis.
    y_bins : array_like
        The bin edges for the y-axis.

    Returns
    -------
    x_repeated : ndarray
        Repeated x-bin edges for the 2D histogram.
    y_repeated : ndarray
        Repeated y-bin edges for the 2D histogram.
    weights : ndarray
        Flattened array of matrix weights for the histogram.
    """
    x_repeated = np.tile(x_bins, len(y_bins))
    y_repeated = np.repeat(y_bins, len(x_bins))
    weights = np.ndarray.flatten(np.transpose(matrix))

    return x_repeated, y_repeated, weights


def get_kinematic_cuts(input_arguments, tof):
    """
    Calculate kinematic cuts based on TOF and input arguments.

    This function computes the minimum and maximum values for S1 and the 
    maximum value for S2 based on the TOF and user-specified cut factors.

    Parameters
    ----------
    input_arguments : list
        Input arguments containing the cut factors or other configuration.
    tof : array_like
        Time of flight bins.

    Returns
    -------
    S1_min : array_like
        The minimum kinematic cut for S1.
    S1_max : array_like
        The maximum kinematic cut for S1.
    S2_max : array_like
        The maximum kinematic cut for S2.
    """
    if '--apply-cut-factors' in input_arguments:
        arg = np.argwhere(input_arguments == '--apply-cut-factors')[0][0]
        c1 = float(input_arguments[arg + 1])
        c2 = float(input_arguments[arg + 2])
        c3 = float(input_arguments[arg + 3])
    else:
        c1 = 1
        c2 = 1
        c3 = 1

    S1_min, S1_max, S2_max = dfs.get_kincut_function(tof, (c1, c2, c3))
    return S1_min, S1_max, S2_max


if __name__ == '__main__':
    pth = ('/home/beriksso/TOFu/analysis/benjamin/other/'
           'ad-firestone/data/100822/100822_49.5_50.0.pickle')
    plot_nes_pickle(pth)
    