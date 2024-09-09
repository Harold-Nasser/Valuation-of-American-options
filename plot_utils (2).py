# from .plot_utils import (mpl, plt, mtick, mdates, gridspec,
#                          set_plt_defaults, set_payoff_axes, set_time_axis, with_style)

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from matplotlib import gridspec

from toolkit import struct

styles = {}
styles['position'] = {'linewidth':1.5, 'color':'b', 'linestyle':'solid'}
styles['position1'] = {'linewidth':1.5, 'color':'b', 'linestyle':'dashed'}
styles['position2'] = {'linewidth':1.5, 'color':'b', 'linestyle':'dotted'}
styles['position3'] = {'linewidth':1.5, 'color':'b', 'linestyle':'dashdot'}
styles['position4'] = {'linewidth':1.5, 'color':'b', 'linestyle':(0, (3, 5, 1, 5, 1, 5))} # 'dashdotdotted'
styles['portfolio'] = {'linewidth':2.00, 'color':'k', 'linestyle':'solid'}
styles['portfolio1'] = {'linewidth':2.00, 'color':'k', 'linestyle':'dashed'}
styles['portfolio2'] = {'linewidth':2.00, 'color':'k', 'linestyle':'dotted'}
styles['gridlines'] = {'linewidth':1.0, 'color':'k', 'linestyle':'dotted'}
with_style = lambda label,name: dict(label=label, **styles[name])

# Inspired from https://www.overleaf.com/learn/latex/Font_sizes%2C_families%2C_and_styles#Reference_guide
def set_plt_defaults(font_small=12, font_normal=14, font_large=16):
    # Save the configuration in fs, for "font size".
    plt.fs = struct(small=font_small, normal=font_normal, large=font_large)
    
    plt.rcParams['font.size'] = str(plt.fs.normal) # Default value for default values?
    plt.rc('font', size=plt.fs.small)              # controls default text sizes
    plt.rc('axes', titlesize=plt.fs.normal)        # fontsize of the axes title
    plt.rc('axes', labelsize=plt.fs.normal)        # fontsize of the x and y labels
    plt.rc('xtick', labelsize=plt.fs.small)        # fontsize of the tick labels
    plt.rc('ytick', labelsize=plt.fs.small)        # fontsize of the tick labels
    plt.rc('legend', fontsize=plt.fs.small)        # legend fontsize
    plt.rc('figure', titlesize=plt.fs.normal)      # fontsize of the figure title
set_plt_defaults()
    
# Payoff vs P&L: In the former, we are not accounting for the prices paid 
#                to enter the position
def set_payoff_axes(ax, xlabel='$S(T)$', ylabel='Payoff'):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    ax.spines[['right','top']].set_visible(False)

    ax.spines['bottom'].set_position('zero')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel(xlabel, horizontalalignment='right', x=1.0)

    ax.spines['left'].set_position(('data',xlim[0]))
    ax.yaxis.set_ticks_position('left')
    ax.set_ylabel('Payoff', horizontalalignment='right', y=1.0) 
    
    # make x-axis arrow
    ax.plot((1), (0), ls="", marker=">", ms=8, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)

    # make y-axis arrow
    ax.plot((xlim[0]), (1), ls="", marker="^", ms=8, color="k",
            transform=ax.get_xaxis_transform(), clip_on=False)

    
def set_time_axis(ax, time_steps, time_labels=None, only=False):    
    """Used successfully in binomial_tree"""
    ax.set_xticks(time_steps)
    if isinstance(time_labels,str):
        ax.set_xticklabels([time_labels%tt for tt in time_steps])
        
    ax.set_xlabel('time', horizontalalignment='right', x=1.0)
    ax.xaxis.set_tick_params(bottom=True, labelbottom=True, top=False)
    if only:
        ax.spines[['top','left','right']].set_visible(False)

    # Add arrows to the spine by drawing triangle shaped point over it
    ax.plot(1, 0, '>k', transform=ax.transAxes, clip_on=False)

    
