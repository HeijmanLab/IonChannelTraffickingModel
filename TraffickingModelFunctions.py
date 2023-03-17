# -*- coding: utf-8 -*-
"""
Author: Stefan Meier
Institute: CARIM Maastricht University
Supervisor: Dr. Jordi Heijman & Prof. Dr. Paul Volders
Date: 24/10/2022
Function: Optimization functions 
"""

# Import the relevant packages
import numpy as np
import pandas as pd 
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from statistics import median 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D  
from matplotlib.gridspec import GridSpec
import seaborn as sns
import myokit
import os

def single_chan(mat, psi, t, dt, nch, factor, n, smooth, mem_save = True, two_state = False, seed = False, plot = False):
    """
    
    This function can be used to simulate the transitions between
    states 0 and 1 at a single channel resolution and with 
    sufficiently small time steps.
    

    Parameters
    ----------
    mat : Array
        Rates matrix.
    
   psi : Float
        Channel production rate.
        
    t : Array
        Time range with step size of dt.
    
    dt : Integer or Float
        Time step size.
    
    nch : Integer
        Number of channels.
    
    factor : Integer
        Divisor array to be used in np.remainder.
    
    n : Integer
        Channel of interest.
        
    smooth : Float or Integer
        Smoothening parameter for KDE plot, where larger values
        represent smoother curves.
    
    mem_save : Boolean, optional
        If True then it uses the factor index to store the transitions
        per channel for computational/memory efficiency. 
        The default is True.
    
    two_state : Boolean, optional
        Whether two states are considered (True) or more (False). 
        The default is False.
    
    seed : Boolean, optional
        Seed to reproduce the random probabilities. 
        The default is False.
    
    plot : Boolean, optional
        Plot the transitions of channel 'n'. 
        The default is False.

    Returns
    -------
    data : Dictionary
        Dictionary with a dataframe that contains the time, the cumulative state distributions 
        and the actual states per channel. Furthermore, the amount of state transitions.

    """

    # Initialize counter
    count = 0
    
    # Initialize an array with channels for lenght nch and state_1
    X = np.ones(nch)
    
    # Determine the amount of states 
    states = len(mat)
    
    # Create an array to store the time, cumulative transitions and the transitions 
    data = np.zeros((int(len(t)/factor), nch + states + 1))
    if mem_save is False:
        data = np.zeros((int(len(t)), nch + states + 1))
    
    # Initialize a new column for data if a new channel is produced
    new_col = np.zeros(len(data))
    
    # Initialize empty arrays to store transitions
    tran_sub = np.zeros(X.shape[0]) # Transition towards sub-membrane
    tran_mem = np.zeros(X.shape[0]) # Transition towards membrane
    tran_non = np.zeros(X.shape[0]) # Transition to non-exisiting state
    tran_any = np.zeros(X.shape[0]) # Transition in any direction
    
    if two_state is True: 
        rows = 3
    else: 
        rows = 4

    # Loop through the time steps and calculate probability of transitioning from state 0 to state 1
    if two_state is True: 
        for i in range(len(t)):
            
            # Uniformal random distribution of probabilities
            if seed is True:
                np.random.seed(i)
                rval = np.random.uniform(size = nch)
            else:
                rval = np.random.uniform(size = nch)
                        
            # Probability matrix because p = rates * dt
            P = mat * dt + np.subtract(np.identity(len(mat)), np.diag(np.matmul(mat, np.ones(len(mat)))) * dt)
             
            # A channel transitions from state 0 to state 1 if the random probability is larger than
            # the probability of staying in state 0
            for j in range(len(rval)):
                if rval[j] > P[int(X[j]), 0]:
                    X[j] = 1
                else:
                    X[j] = 0
                    
            # The data vector saves the amount of channels per state and the states of the first 10 channels
            # This is done for every factor step to save memory
            if mem_save is True:
                if np.remainder(i, factor) == 0:
                    data[count, 0] = t[i]
                    data[count, 1] = sum(X == 0)/nch
                    data[count, 2] = sum(X == 1)/nch
                    data[count, 3:] = (X[0:nch] == 1).reshape(-1, 1).T
                    count += 1
            else:
                data[count, 0] = t[i]
                data[count, 1] = sum(X == 0)/nch
                data[count, 2] = sum(X == 1)/nch
                data[count, 3:] = (X[0:nch] == 1).reshape(-1, 1).T
                count += 1
        
        # Plot the transitions of the channel of interest
        if plot is True:
            plt.figure()
            plt.plot(data[:, 0], data[:, states + n], 'k')
            plt.yticks(np.arange(states), ['Sub-membrane', 'Membrane'])
            plt.xlabel('Time [hrs]')
            plt.title(f'Single channel (#{n}) state transitions over {int(dt*len(t))} hours ')
            plt.tight_layout()
            plt.show()   
         
        # Create a dataframe for storage    
        df = pd.DataFrame(data)
        df = df.rename(columns={0: 'Time', 1: 'State_0', 2: 'State_1'})
    
    if two_state is False: 
        
        # Initialize an array with channels for length nch and state_2
        X = np.full(nch, 2)
        
        # Instead of nch I need to get nM and nS and concatenate them and nS + nM is now nch
        # nS = 50 and nM = 600 for psi * 50
        # Input needs to be an concetenated array of nS and nM
        
        for i in range(len(t)):
            
            # Uniformal random distribution of probabilities
            if seed is True:
                np.random.seed(i)

            # Probability of a channel being created based on binomial distribution,
            # where there are only two possible outcomes for production (succes or failure).
            # If the production of a channel is succesful, then a row needs to be added
            # to the X array and a column to the data array. Furthermore, the 
            # number of channels also increases by one. To prevent the creation and immediate
            # decay of the channel in the same timestep, a delay of one timestep is built-in
            # by the loop_end variable.
            loop_end = nch
            if np.random.binomial(1, (psi * dt)) == 1:
                X = np.append(X, np.ones(1), axis=0)
                tran_sub = np.append(tran_sub, np.ones(1), axis = 0)
                tran_mem = np.append(tran_mem, np.zeros(1), axis = 0)
                tran_non = np.append(tran_non, np.zeros(1), axis = 0)
                tran_any = np.append(tran_any, np.ones(1), axis = 0)
                nch = nch + 1
                data = np.column_stack([data, new_col])

            # Probability matrix because p = rates * dt
            P = mat * dt + np.subtract(np.identity(len(mat)), np.diag(np.matmul(mat, np.ones(len(mat)))) * dt)
            
            # A channel transitions between states if the random probability is larger than
            # than the probability of staying in that state. For this a multinomial distribution
            # is used, which technically is a generalized binomial distribution but the number of
            # possible outcomes is allowed to be greater than 2. In this case, the number of possible
            # outcomes are state 0, state 1 or state 2. 
            for j in range(loop_end):
                tran = np.where(np.random.multinomial(1, P[int(X[j]),:]) == 1)[0]
                # If the current state differs from the new state, then track the transition
                if tran != X[j]:
                    if tran == 1:
                        tran_sub[j] += 1
                    if tran == 2:
                        tran_mem[j] +=1
                    if tran == 0:
                        tran_non[j] +=1
                    tran_any[j] += 1
                X[j] = tran

            # The data vector saves the amount of channels per state and the states of the first 10 channels
            # This is done for every factor step to save memory
            if mem_save is True:
                if np.remainder(i, factor) == 0:
                    data[count, 0] = t[i]
                    data[count, 1] = sum(X == 0)
                    data[count, 2] = sum(X == 1)
                    data[count, 3] = sum(X == 2)
                    data[count, 4:] = (X[0:nch]).reshape(-1, 1).T
                    count += 1
            else:
                data[count, 0] = t[i]
                data[count, 1] = sum(X == 0)
                data[count, 2] = sum(X == 1)
                data[count, 3] = sum(X == 2)
                data[count, 4:] = (X[0:nch]).reshape(-1, 1).T
                count += 1

        # Plot the transitions of the channel of interest
        if plot is True:
            plt.figure()
            plt.plot(data[:, 0], data[:, 3 + n], 'k')
            plt.yticks(np.arange(states), ['Non-existing', 'Sub-membrane', 'Membrane'])
            plt.xlabel('Time [hrs]')
            plt.title(f'Single channel (#{n}) state transitions over {int(dt*len(t))} hours ')
            plt.tight_layout()
            plt.show()  
            
            # Plot the distributions of the transitions
            fig, ax = plt.subplots(rows, 1)
            sns.histplot(tran_any, discrete = True, color = 'blue', stat = 'count', 
                         kde = True, kde_kws = {'bw_adjust' : smooth, 'cut' : 0}, ax = ax[0])
            sns.histplot(tran_mem, discrete = True, color = 'orange', stat = 'count', 
                         kde = True, kde_kws = {'bw_adjust' : smooth, 'cut' : 0}, ax = ax[1])
            sns.histplot(tran_sub, discrete = True, color = 'green', stat = 'count', 
                         kde = True, kde_kws = {'bw_adjust' : smooth, 'cut' : 0}, ax = ax[2])
            if rows > 3:
                if len(np.unique(tran_non)) <= 2 :
                    sns.histplot(tran_non, discrete = True, color = 'purple', stat = 'count', 
                                 kde = False, kde_kws = {'bw_adjust' : smooth, 'cut' : 0}, ax = ax[3])
                else: 
                    sns.histplot(tran_non, discrete = True, color = 'purple', stat = 'count', 
                                 kde = True, kde_kws = {'bw_adjust' : smooth, 'cut' : 0}, ax = ax[3])
            
            # Create a list of the transition arrays and titles for plotting
            kws_list = [tran_any, tran_mem, tran_sub]
            title = ['Transitions in any direction', 'Transitions from sub-membrane to membrane', 'Transitions from membrane to sub-membrane']
            if rows > 3:
                kws_list = [tran_any, tran_mem, tran_sub, tran_non]
                title = ['Transitions in any direction', 'Transitions from sub-membrane to membrane', 
                         'Transitions from membrane to sub-membrane', 'Transitions from sub-membrane to non-exisiting']
            
            # Set the limits and annotate the bars
            for i in range(len(ax)):
                ax[i].set_xlim(0, max(kws_list[i]))
                ax[i].set_ylim(0, (max(np.bincount(kws_list[i].astype(int))) + 200))
                ax[i].set(xlabel = 'Amount of transitions', ylabel = 'Count')
                ax[i].set_title(title[i])
                if i == len(ax) - 1:
                    ax[i].set_xticks(range(len(np.unique(tran_non))))
                for bars in ax[i].containers:
                    ax[i].bar_label(bars)
                ax[i].autoscale(axis = 'x', tight = True)
            fig.tight_layout()

        # Create a dataframe for storage
        df = pd.DataFrame(data)
        df = df.rename(columns={0: 'Time', 1: 'State_0', 2: 'State_1', 3: 'State_2'})
    
    return dict(df = df, tran_mem = tran_mem, tran_sub = tran_sub, tran_non = tran_non, tran_any = tran_any, data = data)

def single_chan_viz(x, interval, hr_list, cmap, bins = True, state_colors = True):
    """
    
    This function can be used to create a dataframe of the transitions per 
    channel over a period of time defined by the length of the hour list 
    at each timestep in the hour list. Subsequently, this can be used to 
    visualize the transitions and states of the channels over time.

    Parameters
    ----------
    x : DataFrame
        DataFrame as outputted by the 'single_chan' function.
    
    interval : Integer
        Select the channels from the total amount of channels at 
        a regular interval. For example, interval = 20 over a 1000 
        channels selects every 50th channel to be displayed.
    
    hr_list : List
        List that contains the indexes of every whole hour.
        
    cmap : List or String
        List that contains the color names of the states
        or a string that defines the Seaborn color palette.
    
    bins: Boolean, optional
        Whether the amount of transitions are binned in categories
        with interval of 5 or not. 
        The default is True.

    state_colors: Boolean, optional
        Determine the colors for states (True) or transitions (False). 
        The default is True.

    Returns
    -------
   Dictionary
        Dictionary that contains one normal dataframe that contains the 
        channels at the interval, a dataframe with the color codes 
        per channel, and a long formatted dataframe that 
        contains the hours, channel number, number of transitions over 
        the past hour, and the state per hour.

    """
    
    # Subset the total amount of channels for visualization
    # based on the interval and the input dataframe's size.
    # Note, that -4 columns are considered because we are only
    # interested in the channels and not in the time and cumulative
    # state distributions.
    chan = x.iloc[:, 4:x.shape[-1]:interval]
    
    # Initialize two dataframes with rows equal to 24 hours and columns equal to the number of channels.
    # This will be used to track the transitions within each hour and the state at each hour. 
    transition = pd.DataFrame(index = range(len(hr_list)), columns = range(chan.shape[1]), dtype = float)
    state = transition.copy()

    # First, loop through the channels and the timesteps (hourly) and track the amount of state 
    # transitions per hour and the state at each hour for each channel, respectively.
    for i in range(chan.shape[1]):
        # (Re)-initialize a transition counter and hour counter for each channel
        tran_count = 0
        hr_count = 0
        for j in range(len(chan)):
            # Track only the hourly timepoints
            if j in hr_list:
                # Create an if-statement to only track the transitions after the initial timepoint
                if j != 0:
                    # If the state of the channel differs between hours than a transition has happen
                    if chan.iloc[j, i] != chan.iloc[j - 1, i]:
                        tran_count += 1
                # Store the transitions and states of the past hour in the dataframes
                transition.iloc[hr_count, i] = tran_count
                state.iloc[hr_count, i] = chan.iloc[j, i]
                hr_count += 1
                # Reset the transition counter after every hour to track the transitions witin each hour
                tran_count = 0
            # Track all the transitions within one hour (e.g. if timestep = 1 min, then 60 steps).
            # After one hour has past then the 'if j in hr_list' statement stores the transition count
            # over the past hour in the transition dataframe.
            elif chan.iloc[j, i] != chan.iloc[j - 1, i]:
                tran_count += 1
        
    # Create a long format transitions dataframe for plotting
    transition = transition.reset_index()
    long_tran = pd.melt(transition, id_vars = 'index', var_name = 'Channel', value_name= 'Transitions (past hr)').astype(int)
    long_tran = long_tran.rename({'index': 'Hours'}, axis = 1)
    if bins is True:
        long_tran['bin'] = 0
    long_tran['color'] = ''
    
    # Create a long format states dataframe for plotting
    state = state.reset_index()
    long_state = pd.melt(state, id_vars = 'index', var_name = 'Channel', value_name= 'State').astype(int)
    long_state = long_state.rename({'index': 'Hours'}, axis = 1) 
    long_state['color'] = ''
    
    # Loop through the transitions or bins and assign a color to each number of transitions or bin
    if state_colors is True:
        tail = state.tail(1)
        col_amount = len(long_state['State'].unique())
        clist = sns.color_palette(cmap, n_colors = col_amount)
        uniq = pd.DataFrame({'color': clist, 'States': np.unique(long_state['State'])})
        long_state = pd.merge(long_state, uniq, left_on = 'State', right_on = 'States', how = 'left')
        long_state = long_state.drop(['States', 'color_x'], axis = 1).rename(columns = {"color_y": "color"})
    else:
        tail = transition.tail(1)
        col_amount = len(long_tran['Transitions (past hr)'].unique())
        clist = sns.color_palette(cmap, n_colors = col_amount)
        uniq = pd.DataFrame({'color': clist, 'Transitions': np.unique(long_tran['Transitions (past hr)'])})
        long_tran = pd.merge(long_tran, uniq, left_on = 'Transitions (past hr)', right_on = 'Transitions', how = 'left')
        long_tran = long_tran.drop(['Transitions', 'color_x'], axis = 1).rename(columns = {"color_y": "color"})
        if bins is True:
            col_bin = np.arange(0, col_amount, 5)
            clist = sns.color_palette(cmap, n_colors = len(np.unique(col_bin)))
            for i, j in enumerate(col_bin):
                long_tran['bin'].loc[long_tran['Transitions (past hr)'] >= j] = j
            for i in range(len(long_tran)):
                long_tran.at[i, 'color'] = clist[np.where(col_bin == long_tran.loc[i, 'bin'])[0][0]]

    # Subset the final row to assign the colors for the lollipop chart   
    tail = tail.drop(tail.columns[0], axis=1).reset_index(drop = True)
    df_colors = pd.DataFrame(0, index = tail.index, columns = tail.columns, dtype = object)
 
    # Loop through the time steps and take the final step of each channel and match it with the color code
    range_tail = np.arange(len(transition)-1 , len(long_tran), len(transition))
    for i in range(len(range_tail)):
        df_colors.at[0, i] = long_tran.loc[range_tail[i], 'color']
        
    # Transform the colors dataframe to long format
    colors_long = pd.melt(df_colors, var_name = 'Channel', value_name= 'Color')  
 
    # Join the long format dataframe together
    tran_state = long_tran.join(long_state['State'])

    return dict(tran_state = tran_state, channels = chan, colors = colors_long, cpalette = clist)

def single_sumplot(x, total_tran, colors, hist_df, cmap, ticks, legend_dict,  work_dir, bins = True, save = True):
    """
    
    Plot a scatterplot of the transitions per channel 
    (defined by interval in 'single_chan_viz') over time 
    and on top of the scatterplot plot a lollipop plot 
    for the total transitions over time for that specific channel.
    It also plots the state occupancy in a histogram.

    Parameters
    ----------
    x : DataFrame
        DataFrame that contains the time, channel, transitions
        over the past hour, state at that hour and total amount
        of transitions over the entire period.
    
    total_tran : DataFrame
        DataFrame that contains the total amount of transitions
        and a specific color tuple which are used for the lollipop 
        plot.
    
    colors : Seaborn ColorPalette
        Seaborn ColorPalette from the function 'single_chan_viz'.
    
    hist_df : DataFrame
        Long-formatted DataFrame that contains the time, the
        state, and the state occupancy per hour.
    
    cmap : List or String
        List that contains the color names of the states
        or a string that defines the Seaborn color palette.
        
    legend_dict : Dictionary
        Dictionary that contains the names of the
        states as keys and the markers as values.   
        
    ticks : Integer
        Interval for the lollipop chart.   
    
    bins: Boolean, optional
        Whether the amount of transitions are binned in categories
        with interval of 5 or not. Same as in 'single_chan_viz'. 
        The default is True.

    Returns
    -------
    A scatterplot, lollipop plot, and histogram.

    """
    
    # Initialize figure and size
    fig = plt.figure(figsize = (16,9))
     
    # Initialize subplots 
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 3])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3], sharey = ax3)
     
    # Use zip to aggregate the elements as tuples to create the lollipop chart
    for t, y, c in zip(total_tran.index, total_tran['Total_tran'], total_tran['Color']):
        ax1.plot([t, t], [0, y], color = c, marker = "o", markevery = (1,2), markersize = 10)
        ax1.set_yticks(np.arange(0, max(total_tran['Total_tran']) + 1, 10))
    ax1.set_xticks([])
    ax1.set_ylabel('Total transitions')
     
    # Create a scatterplot that visualizes the amount of transitions over time for the 
    # selected channels (based on interval)
    if bins is True: 
        sns.scatterplot(data = x, x = 'Channel', y = 'Hours', style = 'State', hue = 'bin',
                        palette = colors, markers = list(legend_dict.values()), ax = ax3, legend = None, s = 100)
    else: 
        sns.scatterplot(data = x, x = 'Channel', y = 'Hours', style = 'State', hue = 'Transitions (past hr)',
                        palette = colors, markers = list(legend_dict.values()), ax = ax3, legend = None, s = 100)
    ax3.set_xlabel('Channels')
    ax3.set_ylabel('Time [hrs]')
    
    # Create a stacked histogram that visualizes the distribution of each state over time
    # for the selected channels (based on interval)
    sns.histplot(data = hist_df, y = 'Time', hue = 'States', palette = cmap, multiple = "stack", 
                 weights = 'Value', discrete = True, ax = ax4, legend = False)
    ax4.tick_params(axis = "y", left = False)
    ax4.set_xticks(np.arange(min(x['Channel']), max(x['Channel']), 2))
    ax4.set_ylabel('')
    ax4.set_xlabel('Channels')
    plt.setp(ax4.get_yticklabels(), visible=False)
    
    # Create a custom legend by first initializing the legends
    legend1 = []
    legend2 = []
    legend3 = []
    
    # Create an overview of the colors that will be used to plot the histogram
    hist_color = sns.color_palette(cmap, n_colors = len(hist_df['States'].unique()))
    
    # Merge the legend_dict with the corresponding colors from hist_color
    list_dict = list(legend_dict)
    dict_hist = {list_dict[0]:hist_color[0], list_dict[1]:hist_color[1],
                 list_dict[2]:hist_color[2]}
    
    # Create the legend labels for the bins
    if bins is True:
        bin_list = np.unique(x['bin'])
        bin_step = bin_list[1] - bin_list[0]
        bin_range = [(bin_list[i], bin_list[i] + (bin_step - 1)) for i in range(len(bin_list))]
    
    # Loop through the colors used in the scatterplot and create a legend
    # for the number of transitions an the states.
    for i,j in enumerate(colors):
        if bins is True:
            legend1.append(Line2D([0], [0], color = 'w', markerfacecolor = j, linewidth = 0, marker = 'o', label = bin_range[i], ms = 15))
        else:
            legend1.append(Line2D([0], [0], color = 'w', markerfacecolor = j, linewidth = 0, marker = 'o', label = i, ms = 15))
        
    for key, value in legend_dict.items():
        legend2.append(Line2D([0], [0], color = 'w', markerfacecolor = 'k', linewidth = 0, marker = value, label = key, ms = 15))
    
    # Loop through the colors used in the histogram and create a legend
    for key, value in dict_hist.items():
        legend3.append(mpatches.Patch(color=value, label=key))
     
    # Number of columns in the legend
    if bins is True:
        ncol_legend = 1
    else:
        ncol_legend = 2

    # Hide the axis on top of the plot and plot the custom legend
    ax2.axis("off")
    first_l = ax2.legend(handles = legend1, loc = "upper left", ncol = ncol_legend, frameon = False, title = 'Transitions')
    second_l = ax2.legend(handles= legend2, loc = 'upper center', frameon = False, title = 'States')
    third_l = ax2.legend(handles = legend3, loc = 'upper right', frameon = False, title = 'States')
    ax2.add_artist(first_l)
    ax2.add_artist(second_l)
    
    # Tidy up the plot. 
    fig.set_tight_layout('tight')
    
    if save is True:
        fig.savefig(work_dir + '\\Figures\\summaryplot.svg', format='svg', dpi=1200)

def diff_eq(y, t, alpha, beta, delta, psi):
    """
    
    The differential function that can be used as input
    for SciPy's 'odeint' to solve system of ordinary differential 
    equations. Computes the derivative of y at t. 

    Parameters
    ----------
    y : List
        List that contains the initial values.
    
    t : Array of integers
        Time range in hours. 
    
    alpha : Float
        Alpha rate per hour (forward trafficking rate).
    
    beta : Float
        Beta rate per hour (backward trafficking rate).
    
    delta : Float
        Delta rate per hour (decay rate).
    
    psi : Float
        Psi rate per hour (production rate).
        
    Returns
    -------
    dydt : Array of floats
        The y-value for every t (solution of dy/dt).

    """
    
    # Divide the y-vector into sub-membrane and membrane
    S, M = y
    
    # The structure of the derivative
    dydt = [psi + beta * M - alpha * S - delta * S, alpha * S - beta * M]
    
    return dydt

def determ_single_chan(mat, arr, nch, t, dt, n, psi, seed = True, plot = True):
    """
    This function is similar to the 'single_chan' function but has been made
    more efficient for optimization purposes.

    Parameters
    ----------
    mat : Array
        Rates matrix.
        
    arr : Array
        Concatenated array of the submembrane (1) and
        membrane (2) occupancy 
        
    nch : Integer
        Total amount of channels in arr.
    
    t : Array
        Time range with step size of dt.
    
    dt : Integer or Float
        Time step size.
        
    n : Integer
        Channel of interest.
     
   psi : Float
        Channel production rate.
    
    seed : Boolean, optional
        Seed to reproduce the random probabilities. 
        The default is False.
    
    plot : Boolean, optional
        Plot the transitions of channel 'n'. 
        The default is False.

    Returns
    -------
    data : Dictionary
        Dictionary with a dataframe that contains the time, the cumulative state distributions 
        and the actual states per channel. Furthermore, the amount of state transitions.

    """

    # Assign the state array to variable X
    X = arr
    
    # Determine the amount of states 
    states = len(mat)
    
    # Initialize counter, rows and smooth
    count, rows, smooth = 0, 4, 1.8
    
    # Create an array to store the time, cumulative transitions and the transitions 
    data = np.zeros((int(len(t)), nch + states + 1))
    
    # Initialize a new column for data if a new channel is produced
    new_col = np.zeros(len(data))
    
    # Initialize empty arrays to store transitions
    tran_sub = np.zeros(X.shape[0]) # Transition towards sub-membrane
    tran_mem = np.zeros(X.shape[0]) # Transition towards membrane
    tran_non = np.zeros(X.shape[0]) # Transition to non-exisiting state
    tran_any = np.zeros(X.shape[0]) # Transition in any direction
    
    for i in range(len(t)):
    
    # Uniformal random distribution of probabilities
        if seed is True:
            np.random.seed(i)
        
        # Probability of a channel being created based on binomial distribution,
        # where there are only two possible outcomes for production (succes or failure).
        # If the production of a channel is succesful, then a row needs to be added
        # to the X array and a column to the data array. Furthermore, the 
        # number of channels also increases by one. To prevent the creation and immediate
        # decay of the channel in the same timestep, a delay of one timestep is built-in
        # by the loop_end variable.
        loop_end = nch
        if np.random.binomial(1, (psi * dt)) == 1:
            X = np.append(X, np.ones(1), axis=0)
            tran_sub = np.append(tran_sub, np.ones(1), axis = 0)
            tran_mem = np.append(tran_mem, np.zeros(1), axis = 0)
            tran_non = np.append(tran_non, np.zeros(1), axis = 0)
            tran_any = np.append(tran_any, np.ones(1), axis = 0)
            nch = nch + 1
            data = np.column_stack([data, new_col])
        
        # Probability matrix because P = rates * dt
        P = mat * dt + np.subtract(np.identity(len(mat)), np.diag(np.matmul(mat, np.ones(len(mat)))) * dt)
        
        # A channel transitions between states if the random probability is larger than
        # than the probability of staying in that state. For this a multinomial distribution
        # is used, which technically is a generalized binomial distribution but the number of
        # possible outcomes is allowed to be greater than 2. In this case, the number of possible
        # outcomes are state 0, state 1 or state 2. 
        for j in range(loop_end):
            tran = np.where(np.random.multinomial(1, P[int(X[j]),:]) == 1)[0]
            # If the current state differs from the new state, then track the transition
            if tran != X[j]:
                if tran == 1:
                    tran_sub[j] += 1
                if tran == 2:
                    tran_mem[j] +=1
                if tran == 0:
                    tran_non[j] +=1
                tran_any[j] += 1
            X[j] = tran
        
        # The data vector saves the amount of channels per state and the states of the first 10 channels
        # This is done for every factor step to save memory
        data[count, 0] = t[i]
        data[count, 1] = sum(X == 0)
        data[count, 2] = sum(X == 1)
        data[count, 3] = sum(X == 2)
        data[count, 4:] = (X[0:nch]).reshape(-1, 1).T
        count += 1
    
    # Plot the transitions of the channel of interest
    if plot is True:
        plt.figure()
        plt.plot(data[:, 0], data[:, 3 + n], 'k')
        plt.yticks(np.arange(states), ['Non-existing', 'Sub-membrane', 'Membrane'])
        plt.xlabel('Time [hrs]')
        plt.title(f'Single channel (#{n}) state transitions over {int(dt*len(t))} hours ')
        plt.tight_layout()
        plt.show()  
        
    # Plot the distributions of the transitions
    fig, ax = plt.subplots(rows, 1)
    sns.histplot(tran_any, discrete = True, color = 'blue', stat = 'count', 
                 kde = True, kde_kws = {'bw_adjust' : smooth, 'cut' : 0}, ax = ax[0])
    sns.histplot(tran_mem, discrete = True, color = 'orange', stat = 'count', 
                 kde = True, kde_kws = {'bw_adjust' : smooth, 'cut' : 0}, ax = ax[1])
    sns.histplot(tran_sub, discrete = True, color = 'green', stat = 'count', 
                 kde = True, kde_kws = {'bw_adjust' : smooth, 'cut' : 0}, ax = ax[2])
    if len(np.unique(tran_non)) <= 2 :
        sns.histplot(tran_non, discrete = True, color = 'purple', stat = 'count', 
                     kde = False, kde_kws = {'bw_adjust' : smooth, 'cut' : 0}, ax = ax[3])
    else: 
        sns.histplot(tran_non, discrete = True, color = 'purple', stat = 'count', 
                     kde = True, kde_kws = {'bw_adjust' : smooth, 'cut' : 0}, ax = ax[3])
    
    # Create a list of the transition arrays and titles for plotting
    kws_list = [tran_any, tran_mem, tran_sub, tran_non]
    title = ['Transitions in any direction', 'Transitions from sub-membrane to membrane', 
             'Transitions from membrane to sub-membrane', 'Transitions from sub-membrane to non-exisiting']
    
    # Set the limits and annotate the bars
    for i in range(len(ax)):
        ax[i].set_xlim(0, max(kws_list[i]))
        ax[i].set_ylim(0, (max(np.bincount(kws_list[i].astype(int))) + 200))
        ax[i].set(xlabel = 'Amount of transitions', ylabel = 'Count')
        ax[i].set_title(title[i])
        if i == len(ax) - 1:
            ax[i].set_xticks(range(len(np.unique(tran_non))))
        for bars in ax[i].containers:
            ax[i].bar_label(bars)
        ax[i].autoscale(axis = 'x', tight = True)
    fig.tight_layout()
    
    # Create a dataframe for storage
    df = pd.DataFrame(data)
    df = df.rename(columns={0: 'Time', 1: 'State_0', 2: 'State_1', 3: 'State_2'})
    
    return dict(df = df, tran_mem = tran_mem, tran_sub = tran_sub, tran_non = tran_non, tran_any = tran_any, data = data)


def sens_analysis(x, model, M, S, scale, hrs, bcl = 3600000):
    '''
    
    This function can be used to perform a senstivity analysis by
    scaling the input parameters.

    Parameters
    ----------
    x : List
        Embedded list with the scaled parameter.
        
    model : Model object of myokit
        Myokit model.
    
    M : Integer
        Amount of channels in the membrane at steady-state.
    
    S : Integer
        Amount of channels in the sub-membrane at steady-state.
    
    scale : List
        List of strings with the scaling factors.
    
    hrs : Integer
        Duration of simulation in hours.
    
    bcl : Integer, optional
        Basic cycle length. The default is 3600000.

    Returns
    -------
    Dictionary
        Dictionary with the simulation outcomes after scaling.

    '''
    
    # Initialize a pacing protocol
    pace = myokit.Protocol()
    
    # Create an event schedule 
    pace.schedule(1, 20, 0.5, bcl, 0) # Used to be 10
    
    # Create a simulation object
    sim = myokit.Simulation(model, pace)
    sim2 = myokit.Simulation(model, pace)
    
    # Set the initial values of M and S
    state = sim.state() 
    new_state = state.copy()
    Mem_index = model.get('ikr_trafficking.M').indice()
    Sub_index = model.get('ikr_trafficking.S').indice()
    new_state[Mem_index] = M
    new_state[Sub_index] = S
    
    # Set the new states
    sim.set_state(new_state)
    sim2.set_state(new_state)
    
    # Initialize dataframes for the state as a function of time
    df_alpha = pd.DataFrame(columns = ['Time', 'M', 'S', 'Scaling', 'Steady_M', 'Steady_S'])
    df_beta = df_alpha.copy()
    df_delta = df_alpha.copy()
    df_psi = df_alpha.copy()
    
    # Initialize counters 
    alpha_counter = 0
    beta_counter = 0
    delta_counter = 0
    psi_counter = 0
    
    # Total duration 
    hr = hrs * bcl
    
    # Loop through the sensitivity list 
    for i in range(len(x)):
        for j in x[i]:
            sim.reset()
            sim.set_state(new_state)
            if i == 0:
                df_alpha_temp = pd.DataFrame(columns = ['Time', 'M', 'S', 'Scaling', 'Steady_M', 'Steady_S'])
                sim.set_constant('ikr_trafficking.a', j[i])
                run = sim.run(hr)
                df_alpha_temp['Time'] = run['engine.time']
                df_alpha_temp['M'] = run['ikr_trafficking.M']
                df_alpha_temp['Steady_M'] = run['ikr_trafficking.M'][-1]
                df_alpha_temp['S'] = run['ikr_trafficking.S']
                df_alpha_temp['Steady_S'] = run['ikr_trafficking.S'][-1]
                df_alpha_temp['Scaling'] = scale[alpha_counter]
                df_alpha = pd.concat([df_alpha, df_alpha_temp])
                alpha_counter += 1
            if i == 1:
                df_beta_temp = pd.DataFrame(columns = ['Time', 'M', 'S', 'Scaling', 'Steady_M', 'Steady_S'])
                sim.set_constant('ikr_trafficking.a', j[i-1])
                sim.set_constant('ikr_trafficking.br', j[i])
                run = sim.run(hr)
                df_beta_temp['Time'] = run['engine.time']
                df_beta_temp['M'] = run['ikr_trafficking.M']
                df_beta_temp['Steady_M'] = run['ikr_trafficking.M'][-1]
                df_beta_temp['S'] = run['ikr_trafficking.S']
                df_beta_temp['Steady_S'] = run['ikr_trafficking.S'][-1]
                df_beta_temp['Scaling'] = scale[beta_counter]
                df_beta = pd.concat([df_beta, df_beta_temp])
                beta_counter += 1
            if i == 2:
                df_delta_temp = pd.DataFrame(columns = ['Time', 'M', 'S', 'Scaling', 'Steady_M', 'Steady_S'])
                sim.set_constant('ikr_trafficking.a', j[i-2])
                sim.set_constant('ikr_trafficking.br', j[i-1])
                sim.set_constant('ikr_trafficking.dr', j[i])
                run = sim.run(hr)
                df_delta_temp['Time'] = run['engine.time']
                df_delta_temp['M'] = run['ikr_trafficking.M']
                df_delta_temp['Steady_M'] = run['ikr_trafficking.M'][-1]
                df_delta_temp['S'] = run['ikr_trafficking.S']
                df_delta_temp['Steady_S'] = run['ikr_trafficking.S'][-1]
                df_delta_temp['Scaling'] = scale[delta_counter]
                df_delta = pd.concat([df_delta, df_delta_temp])
                delta_counter += 1
            if i == 3:
                df_psi_temp = pd.DataFrame(columns = ['Time', 'M', 'S', 'Scaling', 'Steady_M', 'Steady_S'])
                sim.set_constant('ikr_trafficking.a', j[i-3])
                sim.set_constant('ikr_trafficking.br', j[i-2])
                sim.set_constant('ikr_trafficking.dr', j[i-1])
                sim.set_constant('ikr_trafficking.pr', j[i])
                run = sim.run(hr)
                df_psi_temp['Time'] = run['engine.time']
                df_psi_temp['M'] = run['ikr_trafficking.M']
                df_psi_temp['Steady_M'] = run['ikr_trafficking.M'][-1]
                df_psi_temp['S'] = run['ikr_trafficking.S']
                df_psi_temp['Steady_S'] = run['ikr_trafficking.S'][-1]
                df_psi_temp['Scaling'] = scale[psi_counter]
                df_psi = pd.concat([df_psi, df_psi_temp])
                psi_counter += 1
  
    return dict(alpha = df_alpha , beta = df_beta, delta = df_delta, psi = df_psi)

def M_rates(x, factors):
    '''
    
    This function allows you to calculate the change in M
    as a function of the scaled rates. 

    Parameters
    ----------
    x : Dictionary
        Dictionary as outputted by the function 'sens_analysis'.
    
    factors : List
        List of strings with the scaling factors.

    Returns
    -------
    Dictionary
        Dictionary with the relative change in rates with
        respect to 1x scaling.

    '''
    
    a_ref = list()
    b_ref = list()
    d_ref = list()
    p_ref = list()
    
    for i in x:
        ref = x[i].loc[(x[i]['Scaling'] == '1x') & (x[i]['Time'] == 0)]
        M = [x[i].loc[(x[i]['Scaling'] == j) & (x[i]['Time'] == 0)]['Steady_M'].astype(int) for j in factors]
        if i == 'alpha':
            a_ref = [i/np.array(ref['M']) for i in M]
        if i == 'beta':
            b_ref = [i/np.array(ref['M']) for i in M]
        if i == 'delta':
            d_ref = [i/np.array(ref['M']) for i in M]
        if i == 'psi':
            p_ref = [i/np.array(ref['M']) for i in M]
        
    return dict(alpha = a_ref, beta = b_ref, delta = d_ref, psi = p_ref)

def S_rates(x, factors):
    
    '''
    
    This function allows you to calculate the change in S
    as a function of the scaled rates. 

    Parameters
    ----------
    x : Dictionary
        Dictionary as outputted by the function 'sens_analysis'.
    
    factors : List
        List of strings with the scaling factors.

    Returns
    -------
    Dictionary
        Dictionary with the relative change in rates with
        respect to 0% scaling.

    '''
    
    a_ref = list()
    b_ref = list()
    d_ref = list()
    p_ref = list()
    
    for i in x:
        ref = x[i].loc[(x[i]['Scaling'] == '1x') & (x[i]['Time'] == 0)]
        S = [int(x[i].loc[(x[i]['Scaling'] == j) & (x[i]['Time'] == 0)]['Steady_S']) for j in factors]
        if i == 'alpha':
            a_ref = [i/np.array(ref['S']) for i in S]
        if i == 'beta':
            b_ref = [i/np.array(ref['S']) for i in S]
        if i == 'delta':
            d_ref = [i/np.array(ref['S']) for i in S]
        if i == 'psi':
            p_ref = [i/np.array(ref['S']) for i in S]
        
    return dict(alpha = a_ref, beta = b_ref, delta = d_ref, psi = p_ref)

def temp_APsim(x, model, prot, bcl, temp_list, time):
    '''
    
    This function can be used to create/run multiple simulation objects
    with different temperatures.

    Parameters
    ----------
    x : List.
        List with the alpha, beta, delta, and psi rates.
        
    model : MyoKit model object.
        Myokit model.
    
    prot : Myokit protocol
        Myokit pacing protocol.
    
    bcl : Integer
        Basic cycle length.
    
    temp_list : List
        List that contains different temperatures.
    
    time : Integer
        Simulation time in hours.

    Returns
    -------
    Dictionary: 
        Dictionary with the sim_list, first beat and final beat.

    '''
    
    # Initialize a list to store the simulation objects, together with the first and last beat
    sim_list = list()    
    first = list()
    last = list()
    ikr = list()
    apd = list()
    
    # Create a pre-pacing simulation
    sim_pre = myokit.Simulation(model, prot)
    
    # Set the initial values of M and S and temperature
    state_pre = sim_pre.state() 
    new_state_pre = state_pre.copy()
    Mem_index_pre = model.get('ikr_trafficking.M').indice()
    Sub_index_pre = model.get('ikr_trafficking.S').indice()
    new_state_pre[Mem_index_pre] = 2207
    new_state_pre[Sub_index_pre] = 706
    sim_pre.set_state(new_state_pre)
           
    # Set the temperature and the scaling flag, together
    # with the modeltype and the rates
    sim_pre.set_constant('ikr.IKr_modeltype', 1) 
    sim_pre.set_constant('ikr_trafficking.temp_flag', 1)
    sim_pre.set_constant('ikr_MM.IKr_temp', 37)
    sim_pre.set_constant('ikr_trafficking.a', x[0])
    sim_pre.set_constant('ikr_trafficking.br', x[1])
    sim_pre.set_constant('ikr_trafficking.dr', x[2])
    sim_pre.set_constant('ikr_trafficking.pr', x[3])
    
    # Pre-pace for a thousand beats and save the states
    sim_pre.pre(1000 * bcl)
    start_state = sim_pre.state()
    
    # Loop through the temperatures and create the simulation objects
    for i in temp_list:
        
        # Initialize a simulation object
        sim = myokit.Simulation(model, prot)
        
        # Set the initial values of M and S and temperature
        sim.set_state(start_state)
               
        # Set the constants similar as to before
        sim.set_constant('ikr_trafficking.temp_flag', 1)
        sim.set_constant('ikr.IKr_modeltype', 1) 
        sim.set_constant('ikr_trafficking.a', x[0])
        sim.set_constant('ikr_trafficking.br', x[1])
        sim.set_constant('ikr_trafficking.dr', x[2])
        sim.set_constant('ikr_trafficking.pr', x[3])
        sim.set_constant('ikr_MM.IKr_temp', i)
        
        # Append the simulation objects to the list
        sim_list.append(sim)
        
        # Run the simulation
        simulation = sim.run(time * 3600000)
        
        # Split the simulation to each beat
        split = simulation.split_periodic(bcl, adjust = True)
        
        # Append the first and last beat
        first.append(split[0])
        last.append(split[-1])
        
        # Initialize two arrays to store the IKr and APD
        ikr_max_vals = np.zeros(len(split))
        apd_vals = np.zeros(len(split))
        
        # Loop through the splits and store for each beat
        # the IKr and APD
        for j in range(len(split)):
            beat = split[j]
            ikr_max_vals[j] = np.max(beat['ikr.IKr'])
            # Obtain resting membrane potential 
            rmp = np.min(beat['membrane.V']) 
            # Set threshold for APD
            vt = 0.9 * rmp
            # Calculate apd for current beat at current threshold
            curr_apd = beat.apd(threshold=vt, v='membrane.V')
            # Store in list
            if not curr_apd['duration']:
                apd_vals[j] = 0
            else: 
                apd_vals[j] = curr_apd['duration'][0]
            
        
        # Append the IKr and APDs to their respective lists 
        ikr.append(ikr_max_vals)
        apd.append(apd_vals)

        # Always reset the simulation
        sim.reset()
     
    return dict(sim_list = sim_list, first = first, last = last, ikr = ikr, apd = apd)

def APD_IKr_list(sim, before_beats, bcl, Chen = True, apd = True):
    '''
    This function can be used to create a list with either the
    APDs or IKr over time for each temperature (37, 30, 40°C).

    Parameters
    ----------
    sim : Dictionary
        Dictionary as outputted by temp_APsim.
    
    before_beats : Integer
        Amount of beats from steady-state to plot
        the instantenous part.
    
    bcl : Integer
        Basic cycle length.
        
    apd : Boolean, optional
        Add the 27 degrees from Chen et al. 2007. 
        The default is True.
    
    apd : Boolean, optional
        Create an APD list (True) or IKr list (False). 
        The default is True.

    Returns
    -------
    df_list : List
        List with either APDs or IKr per temperature.

    '''
    
    # If-statement to plot either the APD or the IKr
    if apd is True:
        s = sim['apd']
    else:
        s = sim['ikr']

    # Subset the first 10 beats from the steady-state
    # simulation at 37 °C to visualize the instantaneous
    # effect.
    s_steady = s[0][0:(before_beats-1)]
    s_40 = np.concatenate((s_steady, s[1]))
    s_30 = np.concatenate((s_steady, s[2]))
    if Chen is True: 
        s_27 = np.concatenate((s_steady, s[3]))
    
    # Create DataFrames
    if apd is True:
        df_steady = pd.DataFrame(s[0], columns = ['APD'])
        df_40 = pd.DataFrame(s_40, columns = ['APD'])
        df_30 = pd.DataFrame(s_30, columns = ['APD'])
        if Chen is True:
            df_27 = pd.DataFrame(s_27, columns = ['APD'])
    else: 
        df_steady = pd.DataFrame(s[0], columns = ['ikr'])
        df_40 = pd.DataFrame(s_40, columns = ['ikr'])
        df_30 = pd.DataFrame(s_30, columns = ['ikr'])
        if Chen is True:
            df_27 = pd.DataFrame(s_27, columns = ['ikr'])
    
    # Format DataFrames
    df_steady = df_steady.reset_index().rename(columns = {'index':'Time'})
    df_steady['Time'] = df_steady['Time']*bcl/3600000

    df_40 = df_40.reset_index().rename(columns = {'index':'Time'})
    df_40['Time'] = df_40['Time']*bcl/3600000
    
    df_30 = df_30.reset_index().rename(columns = {'index':'Time'})
    df_30['Time'] = df_30['Time']*bcl/3600000
    
    # Create a list with all DataFrames
    df_list = [df_steady, df_40, df_30]
    
    if Chen is True: 
        df_27 = df_27.reset_index().rename(columns = {'index':'Time'})
        df_27['Time'] = df_27['Time']*bcl/3600000
        
        # Create a list with all DataFrames
        df_list = [df_steady, df_40, df_30, df_27]
    
    return df_list
            
def gating_sim(m, vprot, trim, t_tot, temp_list, color, work_dir, save = False):
    '''
    
    This function can be used to show the effects of temperature on gating
    by using a simple two-step voltage clamp protocol 
    (hold = -80 mV, step1 = +20mV, step2 = -30 mV).

    Parameters
    ----------
    m : MyoKit Model object
        Myokit Model. 
    
    vprot : MyoKit Protocol
        MyoKit voltage step protocol.
        
    temp_list : List
        List with temperatures.
    
    color : List
        List with colors.
        
    work_dir: String
        Location of working directory.
    
    save : Boolean, optional
        Save figure in figures folder of directory. The default is False.

    Returns
    -------
    Figure of the effects of temperature on gating.

    '''

    # Set cell type
    m.set_value('cell.mode', 0)

    # Get pacing variable
    p = m.get('engine.pace')

    # Remove binding to pacing mechanism before voltage coupling
    p.set_binding(None)

    # Get membrane potential
    v = m.get('membrane.V')
    # Demote v from a state to an ordinary variable; no longer dynamic
    v.demote()
    # right-hand side setting; value doesn' matter because it gets linked to pacing mechanism
    v.set_rhs(0)
    # Bind v's value to the pacing mechanism
    v.set_binding('pace')

    # Calculate the characterstic time
    tv = vprot.characteristic_time()-1

    # Initialize a list to store the runs and tails
    run_list = list()
    tail_list = list()

    for i in temp_list:
        # Create a voltage step simulation
        sv = myokit.Simulation(m, vprot)
        
        # Set constants for simulation
        sv.set_constant('ikr.IKr_modeltype', 1) 
        sv.set_constant('ikr_MM.IKr_temp', i)
        sv.set_max_step_size(2) 
        
        # Run the simulation protocol and log several variables.
        run = sv.run(tv)
        run_list.append(run)
        
        # Split the log into smaller chunks to get individual steps
        ds = run.split_periodic(t_tot, adjust = True)
        
        # Initialize a list to store the tail values.
        IKr_tail = np.zeros(len(ds))
        
        # Trim each new log to contain the steps of interest by enumerate through 
        # the individual duration steps
        for k, d in enumerate(ds):
            # Adjust the time at the start of every sweep to zero
            steady = run.trim_left(trim[0], adjust = True) 
            # Duration of the peak/steady current, shorter than max duration to prevent interference between steady peak and upslope of tail
            steady = steady.trim_right(trim[1]) 
            # Total step duration (holding potential + peak current + margin of 1ms) to ensure the tail current
            tail = run.trim_left(trim[2], adjust = True) 
            # Duration of the tail current
            tail = tail.trim_right(trim[3])
            # Obtain the peak of the tail
            IKr_tail[k] = max(tail['ikr.IKr'])
        
        # Append the maximum tail values to the list
        tail_list.append(max(IKr_tail))
        
        # Reset simulation 
        sv.reset()
        
    # Visualize the results
    plt.figure(figsize = (6.8, 5))
    plt.subplot(2,1,1)
    for i in range(len(run_list)):
        plt.plot(run_list[i]['engine.time'], run_list[i]['membrane.V'], 'k')
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\bf{V_{M}}$' + ' [mV]')
    plt.subplot(2,1,2)
    for i in range(len(run_list)):
        plt.plot(run_list[i]['engine.time'], run_list[i]['ikr.IKr'], label = f"{temp_list[i]}°C", color = color[i])
    plt.legend()
    plt.xlabel('Time [ms]')
    plt.ylabel(r'$\bf{I_{Kr}}$' + ' [pA/pF]')
    plt.tight_layout()
    if save is True:
        plt.savefig(work_dir + '\\Figures\\voltagestep.svg', format='svg', dpi=1200)
        
    return dict(data = run_list, tail = tail_list)

def mutant_rates(x_arr, t_sim, sec_step, dt, t_determ, y0, intern_df_WT, forward_df_WT, MT_int, MT_fw, target, Kanner_MT_int, Kanner_MT_fw, 
                 Kanner_WT_int, Kanner_WT_fw, int_steady, fw_steady, show = True, plot = True, return_df = False, fw_mt = False):
    '''
    This function can be used to optimize the relative difference between the wildtype (WT) and mutant (MT)
    internalization and forward trafficking rates as seen in Kanner et al. 2018. 

    Parameters
    ----------
    x_arr: List/Array
        Input rates in the following order: alpha, beta, delta, psi.
        
    t_sim : Array
        An array that contains the time range for the stochastic simulation 
        (e.g. much shorter than t_determ).
        
    sec_step : Integer
        The amount that the time step fits in one minute.
        
    dt : Float or Int
        Time step in t for stochastic simulation (e.g. small steps).
        
    t_determ : Array
        An array that contains the time range in hours for the deterministic simulation.
        
    y0 : List
        A list that contains the initial values for both the S and M states.
    
    intern_df_WT : DataFrame
        Wild-type internalization dataframe used as reference. 
   
    forward_df_WT :  DataFrame
        Wild-type forward trafficking dataframe used as reference. 
    
    MT_int : List
        Relative difference in internalization for MT and WT in Kanner et al. 2018.
    
    MT_fw : List
        Relative difference in forward trafficking for MT and WT in Kanner et al. 2018.
        
    target : Integer
        Target number of channels in the membrane for the mutant. 
    
    Kanner_MT_int: DataFrame
        Mutant internalization values for plotting.
        
    Kanner_MT_fw: DataFrame
        Mutant forward trafficking values for plotting.
        
    Kanner_WT_int: DataFrame
        Wildtype internalization values for plotting.
        
    Kanner_WT_fw: DataFrame
        Wildtype forward trafficking values for plotting.
        
    int_steady: Integer
        Amount of channels in the membrane for the reference WT scenario.
        
    fw_steady: Integer
        Amount of channels not in the membrane for the reference WT scenario.
    
   show : Boolean, optional
       Print the errors and corresponding rates. The default is True.
       
   plot : Boolean, optional
       Visualize the results of the optimization. The default is True.
       
   return_df : Boolean, optional
       Return a dictionary with the forward, internalization dataframes, 
       and error values. The default is False.
       
   fw_mt : Boolean, optional
       Only forward mutation, no internalization scaling. The default is False.

    Returns
    -------
    error_tot : Float
       Total error after each iteration

    '''    
    # Subset the rates
    a = x_arr[0]
    b = x_arr[1]
    d = x_arr[2]
    p = x_arr[3]
    
    # Solve the system of ordinary differential equations
    sol_ode = odeint(diff_eq, y0, t_determ, args = (a, b, d, p))
    sol_ode = pd.DataFrame(np.round(sol_ode, 0))

    # Subset the final state distributions to obtain steady-state behaviour
    sub_steady= np.ones(int(sol_ode.iloc[-1][0]))
    mem_steady = np.full(int(sol_ode.iloc[-1][1]), 2)

    # Concatenate the steady state distributions and determine the number of channels
    arr_conc = np.concatenate([sub_steady, mem_steady])
    nch = len(arr_conc)
    
    # Create a new matrix
    mat = np.array([[0, 0, 0], [d, 0, a], [0, b, 0]])

    # Run single_sim simulation 
    MT_sim = determ_single_chan(mat = mat, arr = arr_conc, nch = nch, psi = p, t = t_sim, dt = dt, 
                             n = 500, seed = True, plot = False)
    
    # Subset the dataframe
    single_MT_sim = MT_sim['df']
    
    # Subset the individual channels
    indv_ch = single_MT_sim.iloc[:, 4:]
    
    # Create a list with the indexes at each hour.
    min_list = [int(i) for i in np.arange(0, len(t_sim), sec_step)]
    
    # Check which channels are in membrane state at steady state
    mem_begin = indv_ch.columns[indv_ch.iloc[0] == 2]
    subNE_begin = indv_ch.columns[(indv_ch.iloc[0] == 1) | (indv_ch.iloc[0] == 0)]
    print(f"sub-membrane (begin): {len(subNE_begin)}")
    print(f"membrane (begin): {len(mem_begin)}")
    
    # The internalization was evaluated according to Kanner et al. 2018
    int_10min = single_MT_sim.loc[min_list[10], mem_begin].index[single_MT_sim.loc[min_list[10], mem_begin] != 2]
    int_20min = single_MT_sim.loc[min_list[20], mem_begin].index[single_MT_sim.loc[min_list[20], mem_begin] != 2]
    int_40min = single_MT_sim.loc[min_list[40], mem_begin].index[single_MT_sim.loc[min_list[40], mem_begin] != 2]
    int_60min = single_MT_sim.loc[min_list[60], mem_begin].index[single_MT_sim.loc[min_list[60], mem_begin] != 2]
    
    # The forward trafficking was evaluated according to Kanner et al. 2018
    fw_10min = single_MT_sim.loc[min_list[10], subNE_begin].index[single_MT_sim.loc[min_list[10], subNE_begin] == 2]
    fw_20min = single_MT_sim.loc[min_list[20], subNE_begin].index[single_MT_sim.loc[min_list[20], subNE_begin] == 2]
    fw_40min = single_MT_sim.loc[min_list[40], subNE_begin].index[single_MT_sim.loc[min_list[40], subNE_begin] == 2]
    fw_60min = single_MT_sim.loc[min_list[60], subNE_begin].index[single_MT_sim.loc[min_list[60], subNE_begin] == 2]
    
    intern_df = pd.DataFrame(data = [[0, 0, 'Internalization begin'],
                                    [len(int_10min), 10, 'Internalization for 10 min'],
                                    [len(int_20min), 20, 'Internalization for 20 min'],
                                    [len(int_40min), 40, 'Internalization for 40 min'],
                                    [len(int_60min), 60, 'Internalization for 60 min']],
                             columns = ['Nch', 'Idx', 'Condition'])
    
    forward_df = pd.DataFrame(data = [[0, 0, 'Forward trafficking begin'],
                                     [len(fw_10min), 10, 'Forward trafficking for 10 min'],
                                     [len(fw_20min), 20, 'Forward trafficking for 20 min'],
                                     [len(fw_40min), 40, 'Forward trafficking for 40 min'],
                                     [len(fw_60min), 60, 'Forward trafficking for 60 min']],
                             columns = ['Nch', 'Idx', 'Condition'])
    
    # Incorporate the MT behaviour
    intern_MT_ratio = (((int_steady - intern_df_WT['Nch'][1:])/int_steady)/((len(mem_begin) - intern_df['Nch'][1:])/len(mem_begin))).reset_index(drop = True)
    forward_MT_ratio = (forward_df['Nch'][1:]/forward_df_WT['Nch'][1:]).reset_index(drop = True)
    
    # Print the ratios
    print(f"intern_ratio = {intern_MT_ratio}")
    print(f"forward_ratio = {forward_MT_ratio}")
    print(f"forward rate: {forward_df}")
    print(f"intern rate: {intern_df}")
    
    # If-statement when you want to make it only a forward mutation
    if fw_mt is True:
        MT_int = np.ones(len(MT_int))
        
    # Calculate the SSE for both the itnernalization and forward trafficking 
    error_int = sum([(intern_MT_ratio[i] - MT_int[i])**2 for i in np.arange(len(intern_df.iloc[1:, :]))])
    error_fw = sum([(forward_MT_ratio[i] - MT_fw[i])**2 for i in np.arange(len(forward_df.iloc[1:, :]))])
    
    # Calculate the SSE for the amount of channels in the membrane
    # error_mem = 10 * (1 - len(mem_begin)/1103)**2
    error_mem = 0.00001*(len(mem_begin)-target)**2
    
    # Combine the errors
    error_tot = error_int + error_fw + error_mem
    
    # Plot the results
    if plot is True: 
        plt.figure()
        plt.subplot(1,2,1)
        plt.plot(intern_df['Idx'], (len(mem_begin) - intern_df['Nch'])/len(mem_begin), 'r', label = 'Model MT')
        plt.plot(intern_df['Idx'], (int_steady - intern_df_WT['Nch'])/int_steady, 'b', label = 'Model WT') 
        plt.plot(Kanner_MT_int['x'], Kanner_MT_int['y'], 'or', label = 'Exp. MT')
        plt.plot(Kanner_WT_int['x'], Kanner_WT_int['y'], 'ob', label = 'Exp. WT')
        plt.xlabel('Time [mins]')
        plt.ylabel('Norm. trafficking rate')
        plt.legend()
        plt.title('Internalization')
        
        plt.subplot(1,2,2)
        plt.plot(forward_df['Idx'], forward_df['Nch']/max(forward_df_WT['Nch']), 'r', label = 'Model MT')
        plt.plot(forward_df['Idx'], forward_df_WT['Nch']/max(forward_df_WT['Nch']), 'b', label = 'Model WT') 
        plt.plot(Kanner_MT_fw['x'], Kanner_MT_fw['y'], 'or', label = 'Exp. MT')
        plt.plot(Kanner_WT_fw['x'], Kanner_WT_fw['y'], 'ob', label = 'Exp. WT')
        plt.xlabel('Time [mins]')
        plt.ylabel('Norm. trafficking rate')
        plt.title('Forward trafficking')
        plt.legend()
        plt.tight_layout()
    
    
    if show is True: 
        print ("X: [%f,%f,%f,%f]; Total Errors: %f, Int :%f,  Forward: %f; Mem. Nch: %f"
               % (x_arr[0], x_arr[1], x_arr[2], x_arr[3], error_tot, error_int, error_fw, error_mem))
    
    if return_df is False:    
        return error_tot
    else:
        return dict(intern = intern_df, forward = forward_df, error = error_tot, mem_begin = len(mem_begin))
    
def drug_opt(x, rates, PV, DA, varkevisser, SD_V, asahi, SD_A, TV, TA, TP, y0, work_dir, 
             total_V = False, total_A = False, SD_error = False, save = False, plot = True, 
             show = True, return_df = False):
    '''
    This function can be used to optimize the dependent effects of pentamidine and dofetilide on channel trafficking.
    The data used in this function are from Varkevisser et al. 2013 (doi: 10.1111/bph.12208) and Asahi et al. 2019 
    (doi: 10.1016/j.ejphar.2018.10.046).

    Parameters
    ----------
    x : List
        Parameter list with specific order (hill, amp, km, km', R).
    
    rates : List
        List with alpha, beta, delta, and psi rates.
    
    PV : Integer or Float
        Concentration of pentamidine in Varkevisser et al. 2013 (in uM).
    
    DA : Integer or Float
        Concentration of dofetilide in Asahi et al. 2019 (in uM).
        Note, this cannot be zero due to zero division issues.
    
    varkevisser : DataFrame
        DataFrame with the 'x' column containing the dofetilide concentration
        and 'y' column the percentage of mature hERG channels.
    
    SD_V : DataFrame
        DataFrame with the 'x' column containing the dofetilide concentration
        and 'y' column the standard deviation of the percentage of mature channels.
    
    asahi : DataFrame
        DataFrame with the 'x' column containing the pentamidine concentration
        and 'y' column the percentage of mature hERG channels.
    
    SD_A : DataFrame
        DataFrame with the 'x' column containing the pentamidine concentration
        and 'y' column the standard deviation of the percentage of mature channels.
    
    TV : Integer
        Incubation time for dofetilide in Varkevisser et al. 2013.
    
    TA : Integer
        Incubation time for pentamidine in Asahi et al. 2019.
    
    TP : Integer
        Incubation time for pentamidine in Varkevisser et al. 2013.
    
    y0 : List
        List with the initial sub-membrane and membrane channels.
    
    work_dir : Strings
        Work directory location
        
    total_V : Boolean, optional
        Calculate the total amount of protein (True) or only
        membrane channel (False) for Varkevisser. The default is False.
        
    total_A : Boolean, optional
        Calculate the total amount of protein (True) or only
        membrane channel (False) for Asahi. The default is False.
    
    SD_error: Boolean, optional
        Also take into consideration the SD in the error calculations.
        Default is False. 
          
    save : Boolean, optional
        Save the plots to the designated folder. The default is False.
           
    plot : Boolean, optional
        Visualize the results. The default is True.
    
    show : Boolean, optional
        Show the errors and parameters. The default is True.
        
    return_df : Boolean, optional
        return dataframes instead of error (True). The default is False

    Returns
    -------
    error_tot : Float
        Total sum of squared error between model fit and experimental data.

    '''

    # Subset the parameters
    hill = x[0]
    amp_dof = x[1]
    km_dof = x[2]
    km_prime = x[3]
    R = x[4]
    hillD = x[5]
    b_pent = 6
    hill_pent = 1
   
    # Set the initial condition for the ODE solver
    y0 = y0
    
    # Set the time range for simulation
    TV_hrs = np.arange(TV+1)
    TA_hrs = np.arange(TA+1)
    TP_hrs = np.arange(TP+1)
    TC_hrs = np.arange(24*7+1)
    
    # Initialize lists to store the fractional difference in membrane occupation
    frac_V = list()
    frac_A = list()
    
    # Varkevisser et al. 2013 uses as reference a scenario without dofetilide and pentamidine
    ode_cont = odeint(diff_eq, y0, TC_hrs, args = (rates[0], rates[1], rates[2], rates[3]))
    ode_cont = pd.DataFrame(np.round(ode_cont, 0))
    if total_V is True:
        ref_cont = sum(ode_cont.iloc[-1])
    else: 
        ref_cont = ode_cont.iloc[-1][1]
    
    # To mimic Varkevisser et al. 2013, first incubate with 10 uM pentamidine for 48 hrs.
    # First, given that zero divisions are impossible, a very small number is set as dofetilide.
    small_dof = 1e-15
    L_pre = drug_func(km_prime = km_prime, R = R, hill = hill, amp_dof = amp_dof, km_dof = km_dof, hillD = hillD, D = small_dof, P = PV,
                      b = b_pent, h = hill_pent)
    

    print(ode_cont.iloc[-1])
    print(ode_cont.iloc[-1][0] + ode_cont.iloc[-1][1])

    # Run the deterministic simulation to mimic the Varkevisser's paper
    ode_pre = odeint(diff_eq, y0, TP_hrs, args = (rates[0], rates[1], rates[2], L_pre * rates[3]))
    ode_pre = pd.DataFrame(np.round(ode_pre, 0))
    
    # Store the final state distribution in a new y0 variable and use this as
    # starting point to evaluate the dofetilide treatment
    y0_pre = list(ode_pre.iloc[-1])
    
    # Simulation with dofetilide and without pentamidine
    dof_only = 1
    L_dofonly = drug_func(km_prime = km_prime, R = R, hill = hill, amp_dof = amp_dof, km_dof = km_dof, hillD = hillD, D = dof_only, P = 0,
                      b = b_pent, h = hill_pent)
    # Run the deterministic simulation to mimic the Varkevisser's paper
    ode_dofonly = odeint(diff_eq, y0_pre, TV_hrs, args = (rates[0], rates[1], rates[2], L_dofonly * rates[3]))
    ode_dofonly = pd.DataFrame(np.round(ode_dofonly, 0))
        
    
    print(ode_dofonly.iloc[-1])

    # Loop through the list of dofetilide concentrations from Varkevisser et al. 2013
    for i in varkevisser['x']:
        if i == 0:
            i = small_dof
        
        # Fit the lambda function on the pentamidine and dofetilide concentration from 
        # Varkevisser et al. 2013 Figure 4.
        L_DV = drug_func(km_prime = km_prime, R = R, hill = hill, amp_dof = amp_dof, km_dof = km_dof, hillD = hillD, D = i, P = PV,
                         b = b_pent, h = hill_pent)

        # Run the deterministic simulation to mimic the Varkevisser's paper
        ode_V = odeint(diff_eq, y0_pre, TV_hrs, args = (rates[0], rates[1], rates[2], L_DV * rates[3]))
        ode_V = pd.DataFrame(np.round(ode_V, 0))
        
        if total_V is True:
            # Subset the membrane occupation
            mem_V = np.full(int(ode_V.iloc[-1][1]), 2)
            
            # Subset the submembrane occupation
            sub_V = np.full(int(ode_V.iloc[-1][0]), 1)
            
            # Concatenate the two arrays
            tot_V = np.concatenate((sub_V, mem_V))
               
            # Append the fractional increase/decrease w/r control 
            frac_V.append((len(tot_V)/ref_cont)*100)
            
        else: 
            # Subset the membrane occupation
            mem_V = np.full(int(ode_V.iloc[-1][1]), 2)
            
            # Append the fractional increase/decrease w/r control 
            frac_V.append((len(mem_V)/ref_cont)*100)
                 
    # Calculate the SSE between the model and the Varkevisser data.
    error_V = sum(((frac_V - varkevisser['y'])/varkevisser['y'])**2)
    
    if SD_error is True:
        lb_V = list()
        ub_V = list()

        for i in range(len(varkevisser['y'])):
            lb_V.append(varkevisser.iloc[i, 1] - SD_V.iloc[i, 1])
            ub_V.append(varkevisser.iloc[i, 1] + SD_V.iloc[i, 1])
        
        weight = 100    
        err_V = list()
        
        # # This for-loop adds another factor, where the distance from the datapoint outside the SD to the SD borders
        # # is also taken into account.
        
        for i in range(len(frac_V)):
            if lb_V[i] <= frac_V[i] <= ub_V[i]:
                err_V.append(((frac_V[i] - varkevisser['y'][i])/varkevisser['y'][i])**2)
            elif lb_V[i] < frac_V[i]:
                err_V.append(weight * (abs(lb_V[i] - frac_V[i]) + abs(lb_V[i] - varkevisser['y'][i]))**2)
            else:
                err_V.append(weight * (abs(ub_V[i] - frac_V[i]) + abs(ub_V[i] - varkevisser['y'][i]))**2)
                    
        error_V = sum(err_V)
    
    # Loop through the list of pentamidine concentrations from Asahi et al. 2019
    for i in asahi['x']:
        
        # Fit the lambda function on the pentamidine and dofetilide concentration from
        # Asahi et al. 2019 Figure 1. 
        L_DA = drug_func(km_prime = km_prime, R = R, hill = hill, amp_dof = amp_dof, km_dof = km_dof, hillD = hillD, D = DA, P = i,
                         b = b_pent, h = hill_pent)
        
        # Run the deterministic simulation to mimic the Varkevisser's paper
        ode_A = odeint(diff_eq, y0, TA_hrs, args = (rates[0], rates[1], rates[2], L_DA * rates[3]))
        ode_A = pd.DataFrame(np.round(ode_A, 0))
        
        if total_A is True:
            # Subset the membrane occupation
            mem_A = np.full(int(ode_A.iloc[-1][1]), 2)
            
            # Subset the submembrane occupation
            sub_A = np.full(int(ode_A.iloc[-1][0]), 1)

            # Concatenate the two arrays
            tot_A = np.concatenate((sub_A, mem_A))
            
            # Store the tot_A for the first iteration as ref_A
            if i == 0:
                ref_A = len(tot_A)
            
            # Append the fractional increase/decrease w/r control 
            frac_A.append((len(tot_A)/ref_A)*100)
        
        else: 
            # Subset the sub-membrane and membrane occupation
            mem_A = np.full(int(ode_A.iloc[-1][1]), 2)
            
            # Set the reference amount of channels
            if i == 0:
                ref_A = len(mem_A)
                
            # Append the fractional increase/decrease w/r control 
            frac_A.append((len(mem_A)/ref_A)*100)
            
    # Calculate the SSE between the model and the Asahi data
    error_A = sum(((frac_A - asahi['y'])/asahi['y'])**2)
    
    if SD_error is True:
        lb_A = list()
        ub_A = list()
    
        for i in range(len(asahi['y'])):
            lb_A.append(asahi.iloc[i, 1] - SD_A.iloc[i, 1])
            ub_A.append(asahi.iloc[i, 1] + SD_A.iloc[i, 1])
        
        weight = 100    
        err_A = list()
        
        # This for-loop adds another factor, where the distance from the datapoint outside the SD to the SD borders
        # is also taken into account.
        for i in range(len(frac_A)):
            if lb_A[i] <= frac_A[i] <= ub_A[i]:
                err_A.append(((frac_A[i] - asahi['y'][i])/asahi['y'][i])**2) 
            elif lb_A[i] < frac_A[i]:
                err_A.append(weight * (abs(lb_A[i] - frac_A[i]) + abs(lb_A[i] - asahi['y'][i]))**2)
            else:
                err_A.append(weight * (abs(ub_A[i] - frac_A[i]) + abs(ub_A[i] - asahi['y'][i]))**2)
                
        error_A = sum(err_A)      
            
    # Constrain the optimization by adding a term that penalizes if pentamidine with dofetilide resuls in a larger 
    # lambda than only dofetilide. 
    
    # # Initialize a range of pentamidine and dofetilide concentrations and a scenario without pentamidine
    PENT = np.logspace(-2, 2, 100)
    DOF = np.logspace(-4, 1, 100)
    NO_PENT = 0
    
    # Create a meshgrid for plotting purposes
    PENT, DOF = np.meshgrid(PENT, DOF)
    
    # Intialize a list to store the errors
    err_list = list()
    err_LAMP = list()
    
    # Calculate lambda with dofetilide and pentamidine
    lamda = drug_func(km_prime = km_prime, R = R, hill = hill, amp_dof = amp_dof, km_dof = km_dof, hillD = hillD, D = DOF, P = PENT,
                      b = b_pent, h = hill_pent)

    # Calculate lambda with only dofetilide
    lamda_NP = drug_func(km_prime = km_prime, R = R, hill = hill, amp_dof = amp_dof, km_dof = km_dof, hillD = hillD, D = DOF, P = NO_PENT,
                         b = b_pent, h = hill_pent)

    # Pentamidine with dofetilide needs to be lower than dofetilide by itself
    # and the amplitude of lamda needs to be constraint so that 
    # there cannot be a major increase in ion channels in the WT.
    for i in range(len(lamda.flat)):
        if 1 < lamda_NP.flat[i] < 2.5:
            error_amp = 0
        else:
            error_amp = 10
            
        if lamda.flat[i] < lamda_NP.flat[i]:
            error = 0
        else:
            error = 10
        
        err_list.append(error)
        err_LAMP.append(error_amp)
    
    # Visualize the results
    if plot is True: 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(PENT, DOF, lamda)
        ax.set_xlabel('[Pentamidine]')
        ax.set_ylabel('[Dofetilide]')
        ax.set_zlabel('Lambda')
        
    
    # This cut-off needs to be set, otherwise the temporal relationship will be poor. 
    cut_off = 0.1
    #cut_off = 1e-6
        
    # Add an error term that prevents input parameters with value zero
    error_post = list()
    for i in x:
        if (i == 0) | (i < cut_off):
            error_post.append(100000)
        else:
            error_post.append(0)
    
    # Calculate the errors for all rates being positive
    error_P = sum(error_post)
                
    # Calculate the errors for the lambdas
    error_L = sum(err_list)
    error_L = 0
    
    # Calculate the errors for the amplitude of lambdas
    error_LAMP = sum(err_LAMP)
    error_LAMP = 0
    
    # Calculate the error for the amplitude of alpha
    error_alpha = ((sum(ode_dofonly.iloc[-1])/ref_cont) - 1)**2 #* 10000000
    
    # Calculate the total SSE 
    error_tot = error_A + error_V + error_L + error_LAMP + error_P + error_alpha

    
    if plot is True:
        
        # Create a dataframe for the plot 
        df_A1 = pd.DataFrame({'conc': asahi['x'], 'chan': frac_A, 'error': 0, 'input': 'Model'})
        df_A2 = pd.DataFrame({'conc': asahi['x'], 'chan': asahi['y'], 'error': SD_A['y'], 'input': 'Asahi et al. 2019'})
        df_A = pd.concat([df_A1, df_A2])
        
        df_V1 = pd.DataFrame({'conc': varkevisser['x'], 'chan': frac_V, 'error': 0, 'input': 'Model'})
        df_V2 = pd.DataFrame({'conc': varkevisser['x'], 'chan': varkevisser['y'], 'error': SD_V['y'], 'input': 'Varkevisser et al. 2013'})
        df_V = pd.concat([df_V1, df_V2])
        
        # Plot the results 
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        ax1 = sns.barplot(x = 'conc', y = 'chan', hue = 'input', data = df_A, ax = ax[0], 
                          palette = sns.color_palette('CMRmap_r', n_colors = len(np.unique(df_A['input']))))
        x_c1 = [p.get_x() + 0.5 * p.get_width() for p in ax1.patches]
        y_c1 = [p.get_height() for p in ax1.patches]
        ax[0].legend(loc = 'upper left')
        ax[0].errorbar(x = x_c1, y = y_c1, yerr = df_A["error"], fmt = "none", c = "k")
        ax[0].set_xlabel('Pentamidine [uM]', fontweight = 'bold')
        ax[0].set_ylabel('% in membrane', fontweight = 'bold')
        ax[0].set_title(f'Pentamidine for {TA} hrs', fontweight = 'bold')

        ax2 = sns.barplot(x = 'conc', y = 'chan', hue = 'input', data = df_V, ax = ax[1], 
                          palette = sns.color_palette('CMRmap_r', n_colors = len(np.unique(df_V['input']))))
        x_c2 = [p.get_x() + 0.5 * p.get_width() for p in ax2.patches]
        y_c2 = [p.get_height() for p in ax2.patches]
        ax[1].legend(loc = 'upper left')
        ax[1].errorbar(x = x_c2, y = y_c2, yerr = df_V["error"], fmt = "none", c = "k")
        ax[1].set_xlabel('Dofetilide [uM]', fontweight = 'bold')
        ax[1].set_ylabel('% in membrane', fontweight = 'bold')
        ax[1].set_title(f'Dofetilide for {TV} hrs & Pentamidine [{PV} uM]', fontweight = 'bold')
        fig.tight_layout()
        if save is True:
            fig.savefig(work_dir + '\\Figures\\drug_opt.svg', format = 'svg', dpi = 1200)
        
        
    if show is True:
        print("x: [%f,%f,%f,%f,%f,%f,%f,%f]; Total Errors: %f, Varkevisser :%f, Asahi: %f, Lambda :%f, Amp :%f, alpha :%f, ; Ref_Vark: %f, Ref_Asahi: %f"%
              (hill, amp_dof, km_dof, km_prime, R, hillD, b_pent, hill_pent, error_tot, error_V, error_A, error_L, error_LAMP, error_alpha, ref_cont, ref_A))
    
    if return_df is True:
        return dict(df_V = df_V, df_A = df_A)
    else: 
        return error_tot

def EP_MT(XMT, XWT, model, model_org, scalar, prot, modeltype, bcl, MT_names, work_dir, save = False):
    '''
    This function can be used to visualize the effects of the MT after rates optimization performed 
    by the function 'mutant_rates'. 

    Parameters
    ----------
    XMT : List
        Embedded list with optimized MT rates (alpha, beta, delta, psi). 
        
    XWT : List
        List with optimized WT rates (alpha, beta, delta, psi).
    
    model : Myokit Model object
        Myokit model.
    
    prot : Myokit protocol
        Myokit pacing protocol.
        
    modeltype : Integer
        Value '0' is the original HH IKr model, while '1'
        is the temperature-sensitive Markov model.
        
    bcl : Integer
        Basic cycle length
    
    MT_names : List
        List of strings that represent the scaled rate for that MT behaviour.
        
    work_dir: String
        Location of working directory.
    
    save : Boolean, optional
        Save figure in figures folder of directory. The default is False.
    
    Returns
    -------
    Dictionary with APDs and simulation data. 

    '''
    
    # Initialize a list to store the results
    data_MT_l = list()
    apd_MT_l = list()
    apd_WT_l = list()
    
    # Create a pre-pacing simulation
    sim_WT = myokit.Simulation(model, prot)
    sim_MT = myokit.Simulation(model, prot)
    sim_org = myokit.Simulation(model_org, prot)
    sim_org_down = myokit.Simulation(model_org, prot)
    
    # Set the rates for the WT, prepace, and run the simulation
    sim_WT.set_constant('ikr_trafficking.a', XWT[0])
    sim_WT.set_constant('ikr_trafficking.br', XWT[1])
    sim_WT.set_constant('ikr_trafficking.dr', XWT[2])
    sim_WT.set_constant('ikr_trafficking.pr', XWT[3])
    sim_WT.set_constant('ikr.IKr_modeltype', modeltype) 
    sim_WT.pre(1000 * bcl)
    data_WT = sim_WT.run(bcl, log = ['engine.time', 'ikr_trafficking.S', 'ikr_trafficking.M', 'membrane.V', 'ikr.IKr'])
    
    # Determine the resting membrane potential
    rmp_WT = np.min(data_WT['membrane.V']) 
    # Set threshold for APD
    vt_WT = 0.9 * rmp_WT
    # Calculate apd for current beat at current threshold
    curr_apd_WT = data_WT.apd(threshold = vt_WT, v = 'membrane.V')
    # Store in list
    apd_WT_l.append(curr_apd_WT['duration'][0])
    
    # Set the rates for the original ORD with a scenario with Gkr scaled down.
    sim_org.set_constant('ikr.IKr_modeltype', 0)
    sim_org_down.set_constant('ikr.IKr_modeltype', 0)
    sim_org_down.set_constant('ikr.scalar', scalar)
    sim_org.pre(1000 * bcl)
    sim_org_down.pre(1000 * bcl)
    data_org = sim_org.run(bcl, log = ['engine.time', 'membrane.V', 'ikr.IKr'])
    data_org_down = sim_org_down.run(bcl, log = ['engine.time', 'membrane.V', 'ikr.IKr'])
    
    # Set the rates for the MT, prepace, and run the simulation
    for i in range(len(XMT)):
        sim_MT.set_constant('ikr_trafficking.a', XMT[i][0])
        sim_MT.set_constant('ikr_trafficking.br', XMT[i][1])
        sim_MT.set_constant('ikr_trafficking.dr', XMT[i][2])
        sim_MT.set_constant('ikr_trafficking.pr', XMT[i][3])
        sim_MT.set_constant('ikr.IKr_modeltype', modeltype) 
        sim_MT.pre(1000 * bcl)
        data_MT = sim_MT.run(bcl, log = ['engine.time', 'ikr_trafficking.S', 'ikr_trafficking.M', 'membrane.V', 'ikr.IKr'])
        data_MT_l.append(data_MT)
        
        # Determine the resting membrane potential
        rmp_MT = np.min(data_MT['membrane.V']) 
        # Set threshold for APD
        vt_MT = 0.9 * rmp_MT
        # Calculate apd for current beat at current threshold
        curr_apd_MT = data_MT.apd(threshold = vt_MT, v = 'membrane.V')
        # Store in list
        apd_MT_l.append(curr_apd_MT['duration'][0])
        
        # Reset the simulation
        sim_MT.reset()
        
    # Visualize the action potentials 
    plt.figure()
    plt.plot(data_org['engine.time'], data_org['membrane.V'], label = "Original ORd", color = 'gray', ls = 'dotted')
    plt.plot(data_org_down['engine.time'], data_org_down['membrane.V'], label = f"Original ORd GKr scaled down by {(1-scalar)*100}%", color = 'r', ls = 'dotted')
    plt.plot(data_WT['engine.time'], data_WT['membrane.V'], label = f"WT, Mem. = {max(np.round(data_WT['ikr_trafficking.M']))}", color = 'k')
    for i in range(len(data_MT_l)):
        plt.plot(data_MT_l[i]['engine.time'], data_MT_l[i]['membrane.V'], label = f"{MT_names[i]}, Mem. = {max(np.round(data_MT_l[i]['ikr_trafficking.M']))}", 
                 color = sns.color_palette('CMRmap_r', n_colors = len(MT_names))[i])
    plt.xlabel('Time [ms]')
    plt.ylabel('Membrane potential [mV]')
    plt.legend()
    plt.xlim([0, 600])
    plt.title('Action potential for WT and MT', fontweight = 'bold')
    if save is True: 
        plt.savefig(work_dir + '\\Figures\\MT_EP_AP.svg', format='svg', dpi=1200)
        
    return dict(APD_WT = apd_WT_l, APD_MT = apd_MT_l, data_MT = data_MT_l, data_WT = data_WT, data_org = data_org, data_org_down = data_org_down)

def drug_time(x, rates, model, prot, incub, D, P, T, bcl):
    '''
    This function can be used to mimic the time dependency of the drugs as shown in 
    Varkevisser et al. 2013, where initially the cells were incubated with 10 uM
    Pentamidine for 48 hours, followed by 8 hours of Dofetilide at 1 uM to rescue
    the channels (full rescue). 

    Parameters
    ----------
    x : List
        Parameter list with specific order (hill, amp, km, km', R).
    
    rates : List
        List with alpha, beta, delta, and psi rates.
    
    model : MyoKit model object
        Myokit model.
    
    prot : Myokit protocol
        Myokit pacing protocol.
    
    incub : Integer or Float
        Incubation time in hours.
    
    D : Integer or Float
        Dofetilide concentration in uM
    
    P : Integer or Float
        Pentamidine concentration in uM
    
    T : Integer or Float
        Intervention (dofetilide) time in hours.
    
    bcl : Integer
        Basic cycle length.

    Returns
    -------
    Dictionary
        Dictionary with the simulation data and the maximum IKr amplitude at each beat.

    '''
    
    # Initialize a list to store the results
    ikr = list()
    mem = list()
    sub = list()

    # Create a pre-pacing simulation
    sim_pre = myokit.Simulation(model, prot)
    
    # Set the rates
    sim_pre.set_constant('ikr_trafficking.a', rates[0])
    sim_pre.set_constant('ikr_trafficking.br', rates[1])
    sim_pre.set_constant('ikr_trafficking.dr', rates[2])
    sim_pre.set_constant('ikr_trafficking.pr', rates[3])

    # Set the drug effect parameters
    sim_pre.set_constant('ikr_trafficking.drug_flag', 1)
    sim_pre.set_constant('ikr_trafficking.hill', x[0])      
    sim_pre.set_constant('ikr_trafficking.amp', x[1])      
    sim_pre.set_constant('ikr_trafficking.kmD', x[2])      
    sim_pre.set_constant('ikr_trafficking.kmP', x[3])  
    sim_pre.set_constant('ikr_trafficking.R', x[4])   
    sim_pre.set_constant('ikr_trafficking.hillD', x[5])
    sim_pre.set_constant('ikr_trafficking.b_pent', x[6])
    sim_pre.set_constant('ikr_trafficking.hill_pent', x[7])
    sim_pre.set_constant('ikr.pent_conc', P)
    
    # Pre-pace and save the states
    sim_pre.pre(incub * 3600000)
    start = sim_pre.state()
    
    # Initialize a simulation object
    sim = myokit.Simulation(model, prot)
    
    # Set the initial values after the incubation period
    sim.set_state(start)
           
    # Add the dofetilide treatment and set constants
    sim.set_constant('ikr_trafficking.a', rates[0])
    sim.set_constant('ikr_trafficking.br', rates[1])
    sim.set_constant('ikr_trafficking.dr', rates[2])
    sim.set_constant('ikr_trafficking.pr', rates[3])
    
    sim.set_constant('ikr_trafficking.drug_flag', 1)
    sim.set_constant('ikr_trafficking.hill', x[0])      
    sim.set_constant('ikr_trafficking.amp', x[1])      
    sim.set_constant('ikr_trafficking.kmD', x[2])      
    sim.set_constant('ikr_trafficking.kmP', x[3])  
    sim.set_constant('ikr_trafficking.R', x[4])   
    sim.set_constant('ikr_trafficking.hillD', x[5])
    sim.set_constant('ikr_trafficking.b_pent', x[6])
    sim.set_constant('ikr_trafficking.hill_pent', x[7])
    sim.set_constant('ikr.pent_conc', P)
    sim.set_constant('ikr.dof_conc', D)
    
    # Run the simulation
    data = sim.run(T * 3600000, log = ['engine.time', 'ikr_trafficking.S', 'ikr_trafficking.M', 'membrane.V', 'ikr.IKr'])
    
    # Split the simulation to each beat
    split = data.split_periodic(bcl)
    
    # Initialize two arrays to store the IKr
    ikr_max_vals = np.zeros(len(split))
    m_max = np.zeros(len(split))
    s_max = np.zeros(len(split))
    
    # Loop through the splits and store for each beat the IKr
    for j in range(len(split)):
        beat = split[j]
        ikr_max_vals[j] = np.max(beat['ikr.IKr'])
        m_max[j] = np.max(beat['ikr_trafficking.M'])
        s_max[j] = np.max(beat['ikr_trafficking.S'])
     
    # Append the IKr outcomes to the list    
    ikr.append(ikr_max_vals)
    mem.append(m_max)
    sub.append(s_max)
    
    return dict(data = data, ikr = ikr, split = split, mem = mem, sub = sub)

def Varke_df(X, ref, T):
    '''
    This function can be used to create the data that is suitable
    for plotting the figure as shown in Varkevisser et al. 2013. 

    Parameters
    ----------
    X : Dictionary
        Dictionary as outputted by the function 'drug_time'
    
    ref : Integer
        Reference amount of membrane channels.
        
    T : List
        List with timesteps.

    Returns
    -------
    Dictionary with the amount of membrane, submembrane and total channels.

    '''

    # Subset the amount of channels per hour corresponding to Varkevisser et al. 2013
    mem_vark = list()
    sub_vark = list()
    total_vark = list()
    for i in T:
        if i == 0:
            M = (np.array(X['mem'])[0, i])/ref * 100
            S = (np.array(X['sub'])[0, i])/ref * 100
            T = M + S
        else:
            M = (np.array(X['mem'])[0, i * 60-1])/ref * 100
            S = (np.array(X['sub'])[0, i * 60-1])/ref * 100
            T = M + S
        mem_vark.append(M)
        sub_vark.append(S)
        total_vark.append(T)
        
    return dict(mem = mem_vark, sub = sub_vark, tot = total_vark)

def drug_effects(x, rates, model, prot, D, P,  T, hr_list, bcl, work_dir, title, total = True, plot = True, save = True):
    '''
    This function can be used to simulate and visualize the effects of dofetilide on IKr and channel trafficking.

    Parameters
    ----------
    x : List
        A list that contains the drug parameters.
    
    rates : List
        A list that contains the alpha, beta, delta and psi parameters.
    
    model : MyoKit model
        MyoKit model.
    
    prot : MyoKit protocol
        Pacing protocol.
    
    D : Integer
        Dofetilide concentration in uM
    
    T : Integer
        Time in Hours for the total simulation.
        Note, this amount will be divided by 3 to get
        the simulation time per subsection.
        
    hr_list : List
        A list that contains the timepoints at which the 
        APs will be visualized. 
    
    bcl : Integer
        Basic cycle length.
   
    work_dir : String
        Directory for saving purposes.
        
    title : String
        Name of the file for saving.
    
    total : Boolean, optional
        PLot the total amount of channels (TRUE) or only membrane channels (FALSE).
        The default is True.
   
    plot : Boolean, optional
        Plot the results (TRUE). The default is True.
    
    save : Boolean, optional
        Save the plot (TRUE). The default is True.

    Returns
    -------
    ikr_df : DataFrame
        DataFrame with the current and amount of channels after all the simulations

    '''
    
    # Divide the time (T) by 3 to get the time per simulation
    T = T/3
    
    # Initialize a simulation object
    sim = myokit.Simulation(model, prot)
    
    # Set the rates
    sim.set_constant('ikr_trafficking.drug_flag', 1)
    sim.set_constant('ikr_trafficking.a', rates[0])
    sim.set_constant('ikr_trafficking.br', rates[1])
    sim.set_constant('ikr_trafficking.dr', rates[2])
    sim.set_constant('ikr_trafficking.pr', rates[3])
    sim.set_constant('ikr_trafficking.hill', x[0])      
    sim.set_constant('ikr_trafficking.amp', x[1])      
    sim.set_constant('ikr_trafficking.kmD', x[2])      
    sim.set_constant('ikr_trafficking.kmP', x[3])  
    sim.set_constant('ikr_trafficking.R', x[4])   
    sim.set_constant('ikr_trafficking.hillD', x[5])
    sim.set_constant('ikr_trafficking.b_pent', x[6])
    sim.set_constant('ikr_trafficking.hill_pent', x[7])
    sim.set_constant('ikr.dof_conc', 0)
    sim.set_constant('ikr.pent_conc', P)
    
    # Run the simulation without any drug effects
    d1 = sim.run(T * 3600000, log = ['engine.time', 'ikr_trafficking.S', 'ikr_trafficking.M', 'membrane.V', 'ikr.IKr'])
    
    # Set the drug parameters
    sim.set_constant('ikr.dof_conc', D)
    
    # Run the simulation with drug effects
    d2 = sim.run(T * 3600000, log = d1)
    
    # Remove the drug effects again
    sim.set_constant('ikr.dof_conc', 0)
    
    # Run the recovery simulation
    d3 = sim.run(T * 3600000, log = d2)
    
    # Split the simulation to each beat
    split = d3.split_periodic(bcl, adjust = True)
    
    # Initialize arrays to store the IKr, mem and sub
    ikr = np.zeros(len(split))
    mem = np.zeros(len(split))
    sub = np.zeros(len(split))
    vm = list()
    ms = list()

    # Loop through the splits and store for each beat the IKr
    for j in range(len(split)):
        beat = split[j]
        ikr[j] = np.max(beat['ikr.IKr'])
        mem[j] = round(np.max(beat['ikr_trafficking.M']))
        sub[j] = round(np.max(beat['ikr_trafficking.S']))
        vm.append(beat['membrane.V'])
        ms.append(beat['engine.time'])
 
    # Calculate the total amount of channels
    total_ch = mem + sub
    
    # Create a data dictionary
    data = {'Hours': np.arange(0, sim.time()/3600000, 1),
            'IKr': ikr,
            'Channels': total_ch,
            'Mem': mem,
            'Sub': sub}
    
    # Create a dataframe with the results
    ikr_df = pd.DataFrame(data)
    
    # Initialize a list to store the AP time and membrane potential (mV)
    time = list()
    vM = list()
    
    # Loop through the timepoints and subset the data accordingly 
    for i in hr_list:
        time.append(np.array(ms[i-1]))
        vM.append(np.array(vm[i-1]))
    
    if plot is True:
        # Create two color lists
        sim_colors = sns.color_palette('CMRmap_r', n_colors = 3)
        color_list = sns.color_palette('CMRmap_r', n_colors = len(hr_list))
        
        # Create a grid
        fig = plt.figure(figsize = (8,5))
        gs = GridSpec(nrows = 2, ncols = 2, figure = fig, height_ratios = [1, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
            
        # Plot the IKr over time
        ax1.plot(np.where(ikr_df["Hours"] < 24, ikr_df["IKr"], None), color = sim_colors[0], label = 'Normal')
        ax1.plot(np.where((ikr_df["Hours"] >= 23) & (ikr_df['Hours'] <= 47), ikr_df["IKr"], None), color = sim_colors[1], label= f' {D} uM Dofetilide')
        ax1.plot(np.where(ikr_df["Hours"] >= 47, ikr_df["IKr"], None), color = sim_colors[2], label = "Recovery")
        ax1.set_xlabel('Time [hrs]')
        ax1.set_ylabel('Current density [pA/pF]')
        ax1.set_title('IKr')
        
        # Plot the channel over time
        if total is True: 
            ax2.plot(np.where(ikr_df["Hours"] < 24, ikr_df["Channels"], None), color = sim_colors[0], label = 'Normal')
            ax2.plot(np.where((ikr_df["Hours"] >= 23) & (ikr_df['Hours'] <= 47), ikr_df["Channels"], None), color = sim_colors[1], label= f' {D} uM Dofetilide')
            ax2.plot(np.where(ikr_df["Hours"] >= 47, ikr_df["Channels"], None), color = sim_colors[2], label = "Recovery")
            ax2.set_title('Dofetilide rescue of Kv11.1')
        else:
            ax2.plot(np.where(ikr_df["Hours"] < 24, ikr_df["Mem"], None), color = sim_colors[0], label = 'Normal')
            ax2.plot(np.where((ikr_df["Hours"] >= 23) & (ikr_df['Hours'] <= 47), ikr_df["Mem"], None), color = sim_colors[1], label= f' {D} uM Dofetilide')
            ax2.plot(np.where(ikr_df["Hours"] >= 47, ikr_df["Mem"], None), color = sim_colors[2], label = "Recovery")
            ax2.set_title('Dofetilide rescue of Kv11.1 (mem)')
        ax2.set_xlabel('Time [hrs]')
        ax2.set_ylabel('Number of channels')
        
        # Plot the APs
        for i in range(len(hr_list)):
            ax3.plot(time[i], vM[i], color = color_list[i], label = f'{hr_list[i]}th hour')
        ax3.set_xlim(0, 1000)
        ax3.set_xlabel('Time [ms]')
        ax3.set_ylabel('Membrane potential [mV]')
        ax3.set_title('Action potentials')
        
        # Get the legend object
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax3.get_legend_handles_labels()

        # Delete the axis and plot the legend
        ax4.axis("off")
        first_l = ax4.legend(handles, labels, loc = "upper left", ncol = 1, frameon = False, title = 'Simulation')
        second_l = ax4.legend(handles2, labels2, loc = 'upper right', ncol = 1, frameon = False, title = 'AP timepoints')
        ax4.add_artist(first_l)
    
        # Tidy up the plot
        fig.tight_layout() 
        
        # Save the plpt
        if save is True: 
            fig.savefig(work_dir + "\\Figures\\" + f'{title}' + ".svg", format='svg', dpi=1200)

    return dict(df = ikr_df, time = time, vM = vM)

def drug_func(km_prime, R, hill, amp_dof, km_dof, hillD, D, P, b, h):
    '''
    Function to calculate the scalar for psi to mimic the interplay
    between Pentamidine and Dofetilide. 

    Parameters
    ----------
    km_prime : Int/Float
        km_prime parameter.
        
    R : Int/Float
        R parameter.
    
    hill : Int/Float
        hill coefficient parameter.
    
    amp_dof : Int/Float
        amplitude parameter.
    
    km_dof : Int/Float
        km_dof parameter.
    
    hillD : Int/Float
        hill coefficient for dofetilide parameter.
    
    D : Int/Float
        Dofetilide concentration in uM.
    
    P : Int/Float
        Pentamidine concentration in uM.
    
    b : Int/Float
        Mid-point for pentamidine sigmoid function.
    
    h : Int/Float
        Exponent for pentamidine sigmoid function. 
  

    Returns
    -------
    lamda : Float/Int
        Scalar for psi to mimic drug effects.

    '''
    km = km_prime * (1 + D / R)
    amp_prime = amp_dof/(1 + np.exp((P - b)/-h))
    lamda = (1 / (1 + (P / km)**hill)) * (1 + amp_prime / (1 + (km_dof / D)**hillD))
    return lamda

def APD_original(model, prot, bcl, scalar, labels):
    '''
    This function can be used to determine the baseline behaviour of the
    original ORd, the Markov Model implementation of IKr (Clancy and Rudy), and
    the sensitivity of a reduction in MM IKr. 

    Parameters
    ----------
    model : MyoKit model
        MyoKit model.
    
    prot : MyoKit protocol
        Event schedule.
    
    bcl : Integer
        Basic cycle length.
    
    scalar : Integer or Float
        Scale the IKr current.
    
    labels : List
        List of strings always in the order of
        Original ORd, ORd reduced, MM, MM reduced.

    Returns
    -------
    None.

    '''
    
    # Initialize a lists to store the data and APDs
    APD_list = list()
    data = list()
    color_list = ['k', 'r', 'b', 'green']
    
    # Create two simulation objects for the original ORd and the MM implementation.
    sim_org = myokit.Simulation(model, prot)
    sim_org_d = myokit.Simulation(model, prot)
    sim_MM = myokit.Simulation(model, prot)
    sim_down = myokit.Simulation(model, prot)
    
    # Set the correct modeltypes.
    sim_org.set_constant('ikr.IKr_modeltype', 0)
    sim_org_d.set_constant('ikr.IKr_modeltype', 0)
    sim_org_d.set_constant('ikr.scalar', scalar)
    sim_MM.set_constant('ikr.IKr_modeltype', 1)
    sim_down.set_constant('ikr.IKr_modeltype', 1)
    sim_down.set_constant('ikr.scalar', scalar)
    
    # Create a list of simulations.
    sim_list = [sim_org, sim_org_d, sim_MM, sim_down]
    
    # Loop through the simulations.
    for i in range(len(sim_list)):
        
        # Pre-pace the models
        sim_list[i].pre(1000 * bcl)
        
        # Run the simulations
        data.append(sim_list[i].run(bcl, log = ['engine.time', 'membrane.V', 'ikr.IKr']))
        
        # Determine the resting membrane potential
        rmp = np.min(data[i]['membrane.V'])
        
        # Set APD threshold
        vt = 0.9 * rmp
        
        # Calculate APD for current beat at current threshold
        curr_apd = data[i].apd(threshold = vt, v = 'membrane.V')
        
        # Store in list
        APD_list.append(curr_apd['duration'][0])
        
        
    # Plot the results
    plt.figure()
    for i in range(len(data)):
        plt.plot(data[i]['engine.time'], data[i]['membrane.V'], label = labels[i], color = color_list[i])
    plt.xlabel('Time [ms]')
    plt.ylabel('Membrane potential [mV]')
    plt.legend()
    plt.xlim([0, 600])
    plt.title('Action potential duration', fontweight = 'bold')
    
    return APD_list

def pot_func(alpha_k, km_k, hill_k, scalar_d, conc_k):
    '''
    This function can be used to simulate the effects of hypokalemia 
    by scaling beta and delta.
   

    Parameters
    ----------
    alpha_k : Integer/Float
        Amplitude.
    
    km_k : Integer/Float
        Mid-point.
    
    hill_k : Integer
        Hill coefficient.
    
    scalar_d : Integer/Float
        Relative contribution of potassium to b and d.
   
    conc_k : Integer/Float
        Concentration in mmol/L.

    Returns
    -------
    kb_scalar : Float
        Hypokalemia scalar for beta rate.
    
    kd_scalar : Float
        Hypokalemia scalar for delta rate.

    '''
    # Calculate the scalar to simulate hypokalemia for beta.
    kb_scalar_ref = 1 + ((alpha_k - 1)/(1+(5.4/km_k)**hill_k))
    kb_scalar = 1 + ((alpha_k - 1)/(1+(conc_k/km_k)**hill_k))
    
    # Calculate the scalar to simulate hypokalemia for delta.
    kd_scalar_ref = 1 + ((scalar_d * alpha_k - 1)/(1+(5.4/km_k)**hill_k))
    kd_scalar = 1 + ((scalar_d*alpha_k - 1)/(1+(conc_k/km_k)**hill_k))
    
    return dict(kb = kb_scalar, kd = kd_scalar, kb_ref = kb_scalar_ref, kd_ref = kd_scalar_ref)


def round_off(df, column = 0):
    '''
    This function can be used to round a dataframe column to a whole or half number.

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe
    column : Integer, optional
        Column number The default is 0.

    Returns
    -------
    df : DataFrame
        DataFrame with rounded elements for column of interest.

    '''
    for i in range(len(df)):
        df.iloc[i,column] = round(df.iloc[i, column] * 2)/2
    
    return df

def hypokalemia_opt(x, xfull, inds, rates, incub_hr, recov_hr, incub, recov, conc, conc_week, psi_scale = False, week = False, showit = False, showerror = True, return_df = False):
    '''
    This function can be used to optimize the parameters involved in the hypokalemia simulations through deterministic simulations.

    Parameters
    ----------
    x : List
        List with hypokalemia parameters in the order of alpha, km, hill, scalar.
        
    xfull : array
        Array with all parameters of x.
    
    inds : array
        Array with indices of the parameters of interest in xfull.
        
    rates : List
        List with trafficking parameters (alpha, beta, delta, psi)
    
    incub_hr : Integer
        Incubation time in hours.
    
    recov_hr : Integer
        Recovery time in hours.
    
    incub : DataFrame
        DataFrame with incubation data.
    
    recov : DataFrame
        DataFrame with recovery data.
    
    conc : DataFrame
        DataFrame with concentration-dependence data (overnight incubation).
        
    conc_week : DataFrame
        DataFrame with concentration-dependence data (week incubation).
        
    week : Boolean, optional
        Use the concentration-dependence data after a week. 
    
    showit : Boolean, optional
        Plot the figure. The default is False.
    
    showerror : Boolean, optional
        Print the error terms. The default is True.
        
    return_df : Boolean, optional
        return dataframes instead of error (True). The default is False

    Returns
    -------
    error : Float
        Total error.

    '''
    
    # Set the indices 
    xarr = np.array(xfull)
    xarr[inds] = np.array(x)
    
    # Subset the parameters
    alpha_k = xarr[0]
    km_k = xarr[1]
    hill_k = xarr[2]
    scalar_kd = xarr[3]
    if psi_scale is True:
        psi = xarr[4]
        delta = xarr[5]
      
    # Set the initial condition for the ODE solver.
    y0 = [706, 2209]
    
    # Run a new steady state
    if psi_scale is True:
        steady_time = np.arange(241)
        new_steady = odeint(diff_eq, y0, steady_time, args = (rates[0], rates[1], delta * rates[2],  psi * rates[3]))
        new_steady = pd.DataFrame(np.round(new_steady, 0))
        
        # Subset the number of channels for normalization.
        steady_mem = new_steady.iloc[:, 1]
        steady_sub = new_steady.iloc[:, 0]
        steady_tot = steady_mem + steady_sub
        
        # Define a new y0
        y0 = list(new_steady.iloc[-1])
    
    # Set the time ranges for the simulations.
    incub_hrs = np.arange(incub_hr+1)
    recov_hrs = np.arange(0, recov_hr+0.5, 0.5)
    if week is True:
        week_hrs = np.arange(168+1)
    
    # To mimic Guo et al. 2009, first incubate with 0 mM K+ for 12 hrs (Guo et al. (2009) Figure 3E).
    # First, given that zero divisions are impossible, a very small number is set as K+.
    small_K = 0.1
    pot_pre = pot_func(alpha_k = alpha_k, km_k = km_k, hill_k = hill_k, scalar_d = scalar_kd, conc_k = small_K)
    kb_pre = pot_pre['kb']/pot_pre['kb_ref']
    kd_pre = pot_pre['kd']/pot_pre['kd_ref']
    
    # Run the deterministic simulation and implement the hypoekalemia by scaling beta and delta, respectively.
    if psi_scale is True:
        ode_pre = odeint(diff_eq, y0, incub_hrs, args = (rates[0], kb_pre*rates[1], delta * kd_pre*rates[2], psi * rates[3]))
    else:
        ode_pre = odeint(diff_eq, y0, incub_hrs, args = (rates[0], kb_pre*rates[1], kd_pre*rates[2], rates[3]))
    ode_pre = pd.DataFrame(np.round(ode_pre, 0))
    
    # Store the final state distribution in a new y0 variable and use this as
    #starting point to evaluate the recovery.
    y0_pre = list(ode_pre.iloc[-1])
    
    # Subset the mature channels
    mem_pre = ode_pre.iloc[:, 1]
    sub_pre = ode_pre.iloc[:, 0]
    tot_pre = mem_pre + sub_pre

    # Normalize the membrane channels to baseline mature channels (ca. 2900 channels with original parameter set).
    if psi_scale is True:   
        norm_pre = tot_pre/max(steady_tot)
    else:
        norm_pre = tot_pre/max(tot_pre)

    # Round the time steps of the experimental data for easier indexing.
    incub_round = incub.copy()
    incub_round['x'] = incub_round['x'].round(0)
    incub_round['y'] = incub_round['y']/max(incub_round['y'])
    incub_round['y'] = incub_round['y'] * 100
    
    # Index the correct timesteps based on experimental data.
    pre_index = norm_pre.loc[incub_round['x']].reset_index()
    pre_index.iloc[:, 1] = pre_index.iloc[:, 1] * 100
    
    # Calculate the error after overnight incubation.
    error_decay = sum([(pre_index.iloc[i, 1] - incub_round.iloc[i, 1])**2 for i in range(len(pre_index))])
    
    # Plot the incubation/decay results.
    if showit is True:
        plt.figure()
        plt.plot(incub_round['x'], incub_round['y'], 'k', marker = 'o', label = 'Guo et al. (2009)')
        plt.plot(pre_index['index'], pre_index.iloc[:, 1], 'r', label = 'Model')
        plt.legend()
        plt.xlabel('Time (hrs)')
        plt.xlim([0, 12])
        plt.ylabel('Mature channels (%) relative to baseline') 
        plt.title(f'{incub_hr} hrs incubation with {small_K} mM K+')
        
    # Simulate the recovery (5 mM K+) after 12 hours incubation at low K+ (Guo et al. (2009) Figure 1B).
    pot_recov = pot_func(alpha_k = alpha_k, km_k = km_k, hill_k = hill_k, scalar_d = scalar_kd, conc_k = 5)
    kb_recov = pot_recov['kb']/pot_recov['kb_ref']
    kd_recov = pot_recov['kd']/pot_recov['kd_ref']
    
    # Run the deterministic simulation for the recovery period after incubation.
    if psi_scale is True:
        ode_recov = odeint(diff_eq, y0_pre, recov_hrs, args = (rates[0], kb_recov*rates[1], delta * kd_recov*rates[2], psi * rates[3]))
    else: 
        ode_recov = odeint(diff_eq, y0_pre, recov_hrs, args = (rates[0], kb_recov*rates[1], kd_recov*rates[2], rates[3]))
    ode_recov = pd.DataFrame(np.round(ode_recov, 0))
    
    # Normalize the membrane channels to baseline (ca. 2200 membrane channels with original parameters).
    if psi_scale is True:
        norm_recov = pd.DataFrame(ode_recov.iloc[:, 1]/max(steady_mem))
    else: 
        norm_recov = pd.DataFrame(ode_recov.iloc[:, 1]/ode_pre.iloc[0, 1])
    norm_recov['x'] = norm_recov.index/2
    
    recov_sub = pd.DataFrame(ode_recov.iloc[:, 0])
    recov_sub['x'] = recov_sub.index/2
    recov_mem = pd.DataFrame(ode_recov.iloc[:, 1])
    recov_mem['x'] = recov_mem.index/2
    
    # Round the time steps for easier indexing and normalize to max value.
    recov_round = round_off(recov, column = 0)
    recov_round['y'] = recov_round['y']/max(recov_round['y'])

    # Index the correct timesteps based on experimental data.
    recov_index = pd.merge(norm_recov, recov_round, left_on = 'x', right_on = 'x')
    recov_index.iloc[:, 2] =   recov_index.iloc[:, 2] * 100
    recov_index.iloc[:, 0] =   recov_index.iloc[:, 0] * 100

    # Calculate the recovery error.
    error_recov = sum([(recov_index.iloc[i,0] - recov_index.iloc[i, 2])**2 for i in range(len(recov_index))])

    # Plot the recovery results.
    if showit is True:
        plt.figure()
        plt.plot(recov_index['x'], recov_index.iloc[:,2], 'k', marker = 'o', label = 'Guo et al. (2009)')
        plt.plot(recov_index['x'], recov_index.iloc[:,0], 'r', label = 'Model')
        plt.legend()
        plt.xlabel('Time (hrs)')
        plt.xlim([0, 24])
        plt.ylabel('Membrane channels (%) relative to 12 hours low K+') 
        plt.title(f'Recovery (5 mM) after {incub_hr} hrs low K+')
          
    # Simulate the K+ concentration dependence by looping through the experimental concentrations (Guo et al. (2009) Figure 1C). 
    
    # Initialize a list for storage.
    conc_list = list()
    
    # Loop through the concentrations and run the simulations.
    if week is True:   
        for i in range(len(conc_week)):
            pot_conc = pot_func(alpha_k = alpha_k, km_k = km_k, hill_k = hill_k, scalar_d = scalar_kd, conc_k = conc_week['x'][i])
            kb_conc = pot_conc['kb']/pot_conc['kb_ref']
            kd_conc = pot_conc['kd']/pot_conc['kd_ref']
            
            if psi_scale is True:
                ode_conc = odeint(diff_eq, y0, week_hrs, args = (rates[0], kb_conc*rates[1], delta * kd_conc*rates[2],  psi * rates[3]))
            else: 
                ode_conc = odeint(diff_eq, y0, week_hrs, args = (rates[0], kb_conc*rates[1], kd_conc*rates[2], rates[3]))
            ode_conc = pd.DataFrame(np.round(ode_conc, 0))
            conc_list.append(ode_conc.iloc[-1][1])
    else: 
        for i in range(len(conc)):
            pot_conc = pot_func(alpha_k = alpha_k, km_k = km_k, hill_k = hill_k, scalar_d = scalar_kd, conc_k = conc['x'][i])
            kb_conc = pot_conc['kb']/pot_conc['kb_ref']
            kd_conc = pot_conc['kd']/pot_conc['kd_ref']
            
            if psi_scale is True:
                ode_conc = odeint(diff_eq, y0, incub_hrs, args = (rates[0], kb_conc*rates[1], delta * kd_conc*rates[2], psi * rates[3]))
            else: 
                ode_conc = odeint(diff_eq, y0, incub_hrs, args = (rates[0], kb_conc*rates[1], kd_conc*rates[2], rates[3]))
            ode_conc = pd.DataFrame(np.round(ode_conc, 0))
            conc_list.append(ode_conc.iloc[-1][1])
        
    # Normalize the list to the baseline membrane occupation (ca. 2200 channels with original parameters).
    norm_conc = [(conc_list[i]/conc_list[-1])*100 for i in range(len(conc_list))]

    # Normalize the experimental data.
    if week is True:
        conc_norm_exp = conc_week.copy()
        conc_norm_exp['y'] = conc_norm_exp['y']/conc_norm_exp['y'].iloc[-1]
    else:
        conc_norm_exp = conc.copy()
        conc_norm_exp['y'] = conc_norm_exp['y']/conc_norm_exp['y'].iloc[-1]
    
    # Scale the data to percentage.
    conc_norm_exp['y'] = conc_norm_exp['y'] * 100
        
    # Determine the error for concentration
    if week is True:
        error_conc = sum([(norm_conc[i] - conc_norm_exp['y'][i])**2 for i in range(len(conc_week))])
    else:
        error_conc = sum([(norm_conc[i] - conc_norm_exp['y'][i])**2 for i in range(len(conc))])

    # Plot the concentration dependence
    if showit is True:
        fig, ax = plt.subplots()
        if week is True:
            ax.plot(conc_week['x'], conc_norm_exp['y'], 'k', marker = 'o', label = 'Guo et al. (2009)')
            ax.plot(conc_week['x'], norm_conc, 'r', label = 'Model')
        else:
            ax.plot(conc['x'], conc_norm_exp['y'], 'k', marker = 'o', label = 'Guo et al. (2009)')
            ax.plot(conc['x'], norm_conc, 'r', label = 'Model')
        ax.legend()
        ax.set_xscale('log')
        ax.set_xticks([0.1, 1, 10])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel('K+ concentration (mM)')
        ax.set_ylabel('Membrane channels (%) relative to 20 mmol/L')
        if week is True:
            ax.set_title('Concentration-dependent effects of K+ (week)')
        else:
            ax.set_title('Concentration-dependent effects of K+ (overnight)')
        
    # Sum all the errors.
    error = error_decay + error_recov + error_conc
    
    # Print the error terms.
    if showerror is True:
        print ("X = [%f,%f,%f,%f,%f,%f] # Error: %f, %f, %f, %f"% (xarr[0], xarr[1], xarr[2], xarr[3], xarr[4], xarr[5], error, error_decay, error_recov, error_conc)) 
    
    # Create a DataFrame for the normalized concentration modelling results for easier export.
    norm_conc = pd.DataFrame(norm_conc)
    
    if return_df is True:
        return dict(conc = norm_conc, recov = recov_index, pre = pre_index,
                    conc_exp = conc_norm_exp, pre_exp = incub_round)
    else: 
        return error

def hypok_effects(x, rates, model, prot, T, K_low, K_norm, hr_list, bcl, work_dir, title, total = True, plot = True, save = True):
    '''
    This function can be used to simulate and visualize the effects of hypokalemia on IKr and channel trafficking.

    Parameters
    ----------
    x : List
        A list that contains the hypokalemia parameters.
    
    rates : List
        A list that contains the alpha, beta, delta and psi parameters.
    
    model : MyoKit model
        MyoKit model.
    
    prot : MyoKit protocol
        Pacing protocol.
    
    T : Integer
        Time in Hours for the total simulation.
        Note, this amount will be divided by 3 to get
        the simulation time per subsection.
         
    K_low : Float/Integer
        Low K+ concentration in mmol/L.
        
    K_norm : Float/Integer
        Reference/normal K+ concentration in mmol/L.
    
    hr_list : List
        A list that contains the timepoints at which the 
        APs will be visualized. 
    
    bcl : Integer
        Basic cycle length.
   
    work_dir : String
        Directory for saving purposes.
        
    title : String
        Name of the file for saving.
    
    total : Boolean, optional
        PLot the total amount of channels (TRUE) or only membrane channels (FALSE).
        The default is True.
   
    plot : Boolean, optional
        Plot the results (TRUE). The default is True.
    
    save : Boolean, optional
        Save the plot (TRUE). The default is True.

    Returns
    -------
    Dictionary
        DataFrame with the current and amount of channels after all the simulations

    '''
    
    # Divide the time (T) by 3 to get the time per simulation.
    T = T/3
    
    # Initialize a simulation object
    sim = myokit.Simulation(model, prot)
    
    # Set the rates
    sim.set_constant('ikr_trafficking.a', rates[0])
    sim.set_constant('ikr_trafficking.br', rates[1])
    sim.set_constant('ikr_trafficking.dr', rates[2])
    sim.set_constant('ikr_trafficking.pr', rates[3])
    sim.set_constant('ikr_trafficking.pot_flag', 1)
    sim.set_constant('ikr_trafficking.ak', x[0])
    sim.set_constant('ikr_trafficking.kmk', x[1])
    sim.set_constant('ikr_trafficking.hk', x[2])
    sim.set_constant('ikr_trafficking.scalar_kd', x[3])
    sim.set_constant('extra.Ko', K_norm)
    
    # Pre-pace the simulation.
    sim.pre(1000 * bcl)

    # Run the simulation without any hypokalemic effects.
    d1 = sim.run(T * 3600000, log = ['engine.time', 'ikr_trafficking.S', 'ikr_trafficking.M', 'membrane.V', 'ikr.IKr'])
    
    # Set the hypokalemia.
    sim.set_constant('extra.Ko', K_low)
 
    # Run the simulation with ypokalemia.
    d2 = sim.run(T * 3600000, log = d1)
    
    # Remove the hypokalemia effects again.
    sim.set_constant('extra.Ko', K_norm)
    
    # Run the recovery simulation.
    d3 = sim.run(T * 3600000, log = d2)
    
    # Split the simulation to each beat.
    split = d3.split_periodic(bcl, adjust = True)
    
    # Initialize arrays to store the IKr, mem and sub
    ikr = np.zeros(len(split))
    mem = np.zeros(len(split))
    sub = np.zeros(len(split))
    vm = list()
    ms = list()
    
    # Loop through the splits and store for each beat the IKr
    for j in range(len(split)):
        beat = split[j]
        ikr[j] = np.max(beat['ikr.IKr'])
        mem[j] = round(np.max(beat['ikr_trafficking.M']))
        sub[j] = round(np.max(beat['ikr_trafficking.S']))
        vm.append(beat['membrane.V'])
        ms.append(beat['engine.time'])
 
    # Calculate the total amount of channels
    total_ch = mem + sub
    
    # Create a data dictionary
    data = {'Hours': np.arange(0, sim.time()/3600000, 1),
            'IKr': ikr,
            'Channels': total_ch,
            'Mem': mem,
            'Sub': sub}
    
    # Create a dataframe with the results
    ikr_df = pd.DataFrame(data)
    
    # Initialize a list to store the AP time and membrane potential (mV)
    time = list()
    vM = list()
    
    # Loop through the timepoints and subset the data accordingly 
    for i in hr_list:
        time.append(np.array(ms[i-1]))
        vM.append(np.array(vm[i-1]))
    
    if plot is True:
        # Create two color lists
        sim_colors = sns.color_palette('CMRmap_r', n_colors = 3)
        color_list = sns.color_palette('CMRmap_r', n_colors = len(hr_list))
        
        # Create a grid
        fig = plt.figure(figsize = (8,5))
        gs = GridSpec(nrows = 2, ncols = 2, figure = fig, height_ratios = [1, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
            
        # Plot the IKr over time
        ax1.plot(np.where(ikr_df["Hours"] < 24, ikr_df["IKr"], None), color = sim_colors[0], label = 'Normal')
        ax1.plot(np.where((ikr_df["Hours"] >= 23) & (ikr_df['Hours'] <= 47), ikr_df["IKr"], None), color = sim_colors[1], label= f'{K_low} mmol/L')
        ax1.plot(np.where(ikr_df["Hours"] >= 47, ikr_df["IKr"], None), color = sim_colors[2], label = "Recovery")
        ax1.set_xlabel('Time [hrs]')
        ax1.set_ylabel('Current density [pA/pF]')
        ax1.set_title('IKr')
        
        # Plot the channel over time
        if total is True: 
            ax2.plot(np.where(ikr_df["Hours"] < 24, ikr_df["Channels"], None), color = sim_colors[0], label = 'Normal')
            ax2.plot(np.where((ikr_df["Hours"] >= 23) & (ikr_df['Hours'] <= 47), ikr_df["Channels"], None), color = sim_colors[1], label= f'{K_low} mmol/L')
            ax2.plot(np.where(ikr_df["Hours"] >= 47, ikr_df["Channels"], None), color = sim_colors[2], label = "Recovery")
            ax2.set_title('Hypokalemia on Kv11.1')
        else:
            ax2.plot(np.where(ikr_df["Hours"] < 24, ikr_df["Mem"], None), color = sim_colors[0], label = 'Normal')
            ax2.plot(np.where((ikr_df["Hours"] >= 23) & (ikr_df['Hours'] <= 47), ikr_df["Mem"], None), color = sim_colors[1], label= f'{K_low} mmol/L')
            ax2.plot(np.where(ikr_df["Hours"] >= 47, ikr_df["Mem"], None), color = sim_colors[2], label = "Recovery")
            ax2.set_title('Hypokalemia on Kv11.1 (mem)')
        ax2.set_xlabel('Time [hrs]')
        ax2.set_ylabel('Number of channels')
        
        # Plot the APs
        for i in range(len(hr_list)):
            ax3.plot(time[i], vM[i], color = color_list[i], label = f'{hr_list[i]}th hour')
        ax3.set_xlim(0, 1000)
        ax3.set_xlabel('Time [ms]')
        ax3.set_ylabel('Membrane potential [mV]')
        ax3.set_title('Action potentials')
        
        # Get the legend object
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax3.get_legend_handles_labels()

        # Delete the axis and plot the legend
        ax4.axis("off")
        first_l = ax4.legend(handles, labels, loc = "upper left", ncol = 1, frameon = False, title = 'Simulation')
        second_l = ax4.legend(handles2, labels2, loc = 'upper right', ncol = 1, frameon = False, title = 'AP timepoints')
        ax4.add_artist(first_l)
    
        # Tidy up the plot
        fig.tight_layout() 
        
        return dict(df = ikr_df, time = time, vM = vM)
    
def tran_perhour(x, hr_l):
    '''
    This function calculates the amount of transitions per hour and averages it.

    Parameters
    ----------
    x : DataFrame
        DataFrame that only contains the channels and their states at each timepoint.
   
    hr_l : List
        List with hourly indexes.

    Returns
    -------
    Dictionary
        Dictionary with the amount of transitions per hour,
        average transitions per hour, median transitions per hour,
        and the standard deviation per hour.

    '''
    
    
    # Initialize two dataframes with rows equal to hours and columns equal to the number of channels.
    # This will be used to track the transitions within each hour and the state at each hour. 
    transition = pd.DataFrame(index = range(len(hr_l)), columns = range(x.shape[1]), dtype = float)
    state = transition.copy()
    
    # First, loop through the channels and the timesteps (hourly) and track the amount of state 
    # transitions per hour and the state at each hour for each channel, respectively.
    for i in range(x.shape[1]):
        # (Re)-initialize a transition counter and hour counter for each channel
        tran_count = 0
        hr_count = 0
        for j in range(len(x)):
            # Track only the hourly timepoints
            if j in hr_l:
                # Create an if-statement to only track the transitions after the initial timepoint
                if j != 0:
                    # If the state of the channel differs between hours than a transition has happen
                    if x.iloc[j, i] != x.iloc[j - 1, i]:
                        tran_count += 1
                # Store the transitions and states of the past hour in the dataframes
                transition.iloc[hr_count, i] = tran_count
                state.iloc[hr_count, i] = x.iloc[j, i]
                hr_count += 1
                # Reset the transition counter after every hour to track the transitions witin each hour
                tran_count = 0
            # Track all the transitions within one hour (e.g. if timestep = 1 min, then 60 steps).
            # After one hour has past then the 'if j in hr_list' statement stores the transition count
            # over the past hour in the transition dataframe.
            elif x.iloc[j, i] != x.iloc[j - 1, i]:
                tran_count += 1
    
    # Subset the transitions after the initial starting point
    transitions = transition.iloc[1:, :].reset_index(drop = True)
    
    # Only calculate the average amount of transitions of a channel that 
    # actually exists (e.g. not decayed), so change 0 transitions to NaN.
    transitions = transitions.replace({0 : np.nan})
    
    # Calculate the average, median and standard deviation per hour
    trans_avg = transitions.mean(axis = 1, skipna = True)
    trans_med = transitions.median(axis = 1, skipna = True)
    trans_std = transitions.std(axis = 1, skipna = True)
    
    return dict(avg = trans_avg, median = trans_med, std = trans_std, overview = transitions)

def int_rec_plot(X, title, only_perc = False):
    """
    
    A function to plot the difference between the amount of internalized channels and recycled channels.
    

    Parameters
    ----------
    X : Dataframe
        Single channel simulation dataframe consisting of 'Nch' column with
        number of channels per condition, 'Idx' column to plot the x-coordinates, 
        and 'Condition' that is either Internalization or Recycling.
    
    title : String
        Title of the plot.
    
    only_perc : Boolean. Optional.
        False is plot the value and return percentages. True
        is only return percentages. 

    Returns
    -------
    Figure and the fractional difference between the states.

    """
    
    if only_perc is True:
        # Initialize a counter
        count = 0  
        
        # Initialize an array to store the fractional difference
        frac = np.zeros(len(X) - 1)
        
        # Loop through the sorted dataframe to obtain the y-coordinates and height of the
        # text to plot the percentage difference between the internalized and recycled portion
        # on top of the bar plot. 
        for i in range(1, len(X)):
            y =  max(X['Nch'])
            ymin = X['Nch'][count + 1]
            perc = round((ymin / y * 100), 1)
            frac[i - 1] = round((ymin / y), 3)
            count +=1
    else:     
    
        # Initialize a figure
        plt.figure()
        
        # Set color and create a barplot
        cmap = sns.color_palette('colorblind')
        sns.barplot(data = X, x = 'Idx', y = 'Nch', hue = 'Condition', palette = cmap)
        
        # Create an ticks object and fill the ticks to have an equal distribution
        # of te bars
        n_points = len(X)
        xticks = np.zeros(n_points)
        
        # If the number of points is uneven then add one to the negative side
        if n_points % 2 == 0:
            start_point = round(0.1 - 0.1 * n_points, 1)
        else:
            start_point = round(-0.1 * n_points, 1)
    
        # Loop through the number of points and assign their x-coordinates
        for i in range(n_points):
            if i == 0:
                xticks[i] = start_point 
            else:
                xticks[i] = round(xticks[i - 1] + 0.2, 1)
        
        # Initialize a counter
        count = 0     
        
        # Initialize an array to store the fractional difference
        frac = np.zeros(len(X) - 1)
        
        # Loop through the sorted dataframe to obtain the y-coordinates and height of the
        # text to plot the percentage difference between the internalized and recycled portion
        # on top of the bar plot. 
        for i in range(1, len(X)):
            y, h, col = max(X['Nch']), 2, 'k'
            if y > 100:
                y, h, col = max(X['Nch']), 20, 'k'
            ymin = X['Nch'][count + 1]
            perc = round((ymin / y * 100), 1)
            frac[i - 1] = round((ymin / y), 3)
            plt.plot([xticks[count], xticks[count], xticks[count + 1] , xticks[count + 1]], [y, y + h, y + h, ymin], lw = 1.5, c = col)
            plt.text((xticks[count] + xticks[count + 1])*.5, y + h, f'{perc}%', ha = 'center', va = 'bottom', color = cmap[i])
            count +=1
        
        # Plot labels and title
        plt.xticks([])
        plt.xlabel('Time [min]')
        plt.ylabel('Number of channels')
        plt.legend()
        plt.title(title)
        plt.tight_layout()
    
    return frac

def determ_opt(rate_arr, y0, t_determ, t, dt, n, cont_int, cont_HeLa, cont_H9C2, cont_rec, cont_recA,
               min_index, min_step, HeLa = True, show = True):
    """
    
    This function can be used to optimize the rates of the stochastic model to minimize the 
    sum of squared errorsbetween the model's internalization and recycling rates and those 
    from lierature (Dennis et al. 2011 doi:10.1074/jbc.M111.254367). 

    Parameters
    ----------
    rate_arr : List
        A list that contains the alpha, beta, delta, and psi rate (in that specific order).
   
    y0 : List
        A list that contains the initial values for both the S and M states.
    
    t_determ : Array
        An array that contains the time range in hours for the deterministic simulation.
    
    t : Array
        An array that contains the time range for the stochastic simulation 
        (e.g. much shorter than t_determ).
    
    dt : Float or Int
        Time step in t for stochastic simulation (e.g. small steps).
    
    n : Int
        Plot the transitions of the nth channel.
    
    cont_int : DataFrame
        A DataFrame that contains the time and fraction of internalized channels is used to optimize 
        the internalization rate.
    
    cont_rec : DataFrame
        A DataFrame that contains the time and fraction of recycled channels is used to optimize 
        the recycling rate.
    
    cont_recA : DataFrame
        A DataFrame that contains the time and fraction of recycled channels is used to optimize 
        the recycling rate (Apaja et al. 2013).
    
    min_index : List
        A list that contains the indexes of every minute (given the dt).
    
    min_step : Integer
        The amount that the time step fits in one minute.
    
    HeLa: Boolean, optional
        Calculate the internalization error with HeLa or H9C2. The default is True.

    show : Boolean, optional
        Print the errors and corresponding rates. The default is True.

    Returns
    -------
    error_tot : Float
       Total error after each iteration

    """
    
    # Subset the alpha, beta, delta, psi rates (per hour)
    a = rate_arr[0]
    b = rate_arr[1]
    d = rate_arr[2]
    p = rate_arr[3]
    
    # Solve the system of ordinary differential equations
    sol_ode = odeint(diff_eq, y0, t_determ, args = (a, b, d, p))
    sol_ode = pd.DataFrame(np.round(sol_ode, 0))

    # Subset the final state distributions to obtain steady-state behaviour
    sub_steady= np.ones(int(sol_ode.iloc[-1][0]))
    mem_steady = np.full(int(sol_ode.iloc[-1][1]), 2)

    # Concatenate the steady state distributions and determine the number of channels
    arr_conc = np.concatenate([sub_steady, mem_steady])
    nch = len(arr_conc)
    print(f"sub-membrane (begin): {len(sub_steady)}")
    print(f"membrane (begin): {len(mem_steady)}")
    
    # Create the rate matrix
    mat = np.array([[0, 0, 0], [d, 0, a], [0, b, 0]])
    
    # Run the single cell simulation
    determ_sim_dict = determ_single_chan(mat = mat, arr = arr_conc, nch = nch, psi = p, t = t, dt = dt, n = n, seed = True, plot = True)

    # Subset the dataframe
    determ_df = determ_sim_dict['df']
    
    # Subset the individual channels
    indv_ch = determ_df.iloc[:, 4:]
    
    # Create a list with the indexes at each hour.
    hr_list = [int(i) for i in np.arange(0, len(determ_df), min_step*60)]
    
    # Calculate the average amount of transitions per hour
    avg_tran = tran_perhour(indv_ch, hr_list)
    print(f"mean transitions {avg_tran['avg'].mean()}")
    
    # Check which channels are in membrane state at steady state
    mem_begin = indv_ch.columns[indv_ch.iloc[0] == 2]
    
    # The internalization was evaluated according to Dennis et al. 2011 (fig 6B)
    int_5min = determ_df.loc[min_index[5], mem_begin].index[determ_df.loc[min_index[5], mem_begin] == 1]
    int_30min = determ_df.loc[min_index[30], mem_begin].index[determ_df.loc[min_index[30], mem_begin] == 1]
    int_60min = determ_df.loc[min_index[60], mem_begin].index[determ_df.loc[min_index[60], mem_begin] == 1]
    
    # The internalization was also evaluated according to Apaja et al. 2013 (Fig 4)
    int_90min = determ_df.loc[min_index[90], mem_begin].index[determ_df.loc[min_index[90], mem_begin] == 1]
    
    # Recycling after internalization on the basis of Dennis et al. 2011 (fig 6C)
    # Note, the methods state that they first incubated for 30 mins and then evaluated recycling
    # Subset the amount of channels that were internalized after 30 mins of incubation
    incub_30min = int_30min.copy()
    
    # Determine the recycled channels
    rec_3min = determ_df.loc[min_index[33], incub_30min].index[determ_df.loc[min_index[33], incub_30min] == 2]
    rec_10min = determ_df.loc[min_index[40], incub_30min].index[determ_df.loc[min_index[40], incub_30min] == 2]
    rec_20min = determ_df.loc[min_index[50], incub_30min].index[determ_df.loc[min_index[50], incub_30min] == 2]
    
    mem_end = indv_ch.columns[indv_ch.iloc[-1] == 2]
    sub_end = indv_ch.columns[indv_ch.iloc[-1] == 1]
    print(f"membrane (end) {len(mem_end)}")
    print(f"sub-membrane (end) {len(sub_end)}")
    
    # Create a df to plot the internalization
    intern_df = pd.DataFrame(data = [[len(mem_begin), 1, 'Steady-state membrane'],
                                     [len(int_5min), 1, 'Internalization for 5 min'],
                                     [len(int_30min), 1, 'Internalization for 30 min'],
                                     [len(int_60min), 1, 'Internalization for 60 min'],
                                     [len(int_90min), 1, 'Internalization for 90 min']],
                             columns = ['Nch', 'Idx', 'Condition'])
    
    # Plot the internalization
    intern_Dennis = int_rec_plot(intern_df, 'Internalized channels based on Dennis et al. 2011', only_perc = True) * 100
    print(f"Model internalization: {intern_Dennis}")
    
    # Create a df to plot the recycling
    recycle_df = pd.DataFrame(data = [[len(incub_30min), 1, '30 min internalization'],
                                      [len(rec_3min), 1, 'Recycling for 3 min'],
                                      [len(rec_10min), 1, 'Recycling for 10 min'],
                                      [len(rec_20min), 1, 'Recycling for 20 min']],
                              columns = ['Nch', 'Idx', 'Condition'])

    # Plot the recycling
    recycle_Dennis = int_rec_plot(recycle_df, 'Recycled channels based on Dennis et al. 2011', only_perc = True) * 100
    print(f"Model recycling: {recycle_Dennis}")
    
    
    # Minimize the error with the expected amount of IKr channels in the membrane (i.e., 2000) 
    # based on Heijman et al. 2013 supplementary, doi: https://doi.org/10.1371/journal.pcbi.1003202
    error_nch = 0.1 * (2200 - len(mem_begin))**2
    
    # Optimize internalization and recycling separately by calculating separate error terms
    error_int = list()
    for i in range(len(intern_Dennis)):
        if i < 3:
            error_int.append((intern_Dennis[i] - cont_int.iloc[i, 1])**2)
        if i == 3:
            if HeLa is True:
                error_int.append((intern_Dennis[i] - cont_HeLa.iloc[0, 1])**2)
            else: 
                error_int.append((intern_Dennis[i] - cont_H9C2.iloc[0, 1])**2)
    error_int = sum(error_int)
    
    # Recycling error
    rec_error1 = list()
    rec_error2 = list()
    for i in range(len(recycle_Dennis)):
        rec_error1.append((recycle_Dennis[i] - cont_rec.iloc[i, 1])**2)
        if i == 1:
            rec_error2.append((recycle_Dennis[i] - cont_recA.iloc[0, 1])**2)
        if i == 2:
            rec_error2.append((recycle_Dennis[i] - cont_recA.iloc[3, 1])**2)
    error_rec = sum(rec_error1) + sum(rec_error2)
    
    # Minimize the amount of transitions relative to the (average) amount of transitions per hour
    # (i.e. < 6) as shown in Ghosh et al. 2018 Fig 2, doi: 10.1016/j.bbamcr.2018.06.
    # However, because of the normal distribution an average transitions per hour was set to 4.
    error_tran = 100 * sum([(4 - avg_tran['avg'][i])**2 for i in range(len(avg_tran['avg']))])

    # Sum the errors
    error_tot = error_int + error_rec + error_nch + error_tran
    
    if show is True: 
        print ("X: [%f,%f,%f,%f]; Int rate: %f, %f, %f ;Rec rate: %f, %f, %f, Total Errors: %f, %f, %f, %f, %f; Mem. Size: %f"
               % (rate_arr[0], rate_arr[1], rate_arr[2], rate_arr[3], intern_Dennis[0], intern_Dennis[1],
                  intern_Dennis[2], recycle_Dennis[0], recycle_Dennis[1], recycle_Dennis[2],
                  error_tot, error_int, error_rec, error_nch, error_tran, len(mem_begin)))
               
    return dict(error = error_tot, recycle = recycle_df, rec_list = recycle_Dennis, int_list = intern_Dennis, intern = intern_df)

def graphpad_exp(d, threshold):
    '''
    This function can be used to format the drug data for export to GraphPad.

    Parameters
    ----------
    d : Dictionary
        Dictionary of numpy arrays as exported by 'drug_effects' as keys 'time' and 'vM.

    threshold : Integer
        Time cut-off in ms (1000 = all the data up to 1000 ms index). 

    Returns
    -------
    List with DataFrames ready to be exported to GraphPad.

    '''
    export = list()
    for i in range(len(d['time'])):
                pos = np.searchsorted(d['time'][i], threshold, side='right') + 1
                time = pd.DataFrame(d['time'][i][:pos])
                vm = pd.DataFrame(d['vM'][i][:pos])
                merged = time.merge(vm, left_index=True, right_index=True, how='inner')
                merged.rename(columns={'0_x':'time', '0_y':'vm'}, inplace=True)
                export.append(merged)
    
    return export

def mono_exp(x, a, tau, c):
    """ Mono-exponential function to calculate the rates
    """
    return a * np.exp(-x/tau) + c


def double_exp(x, a1, tau1, a2, tau2, c):
    """ Double-exponential function to calculate the rates
    """
    return a1 * np.exp(-x/tau1) + a2 * np.exp(-x/tau2) + c

def Q10(a1, a2, t1, t2):
    """Q10 function 
    The Q10 function calculates the Q10 coefficients which can be used
    for temperature correction. The Q10 coefficient is the change in conductance
    or rate for for each 10°C change in temperature.
    
    Parameters
    ----------
    a1 : Float/Integer
        Rate at the first temperature (t1) (e.g. activation rate at t1). 
        
    a2 : Float/Integer
         Rate at the second temperature (t2) (e.g. activation rate at t2).
         
    t1 : Float/Integer
         First temperature (e.g. room temperature 22°C).
    
    t2 : Float/Integer
        Second temperature (e.g. physiological temperature 37°C).

    Returns
    -------
    Q10 value.
    
    """
    q10 = (a1 / a2)**(10.0/(t2 - t1))
    return q10
    
def ZhouActTime(modeltype, temp, sim, time, tot_dur, hold, t_steps, showit = 0, showcurve = 0, log = 0):
    """ ZhouActTime function
    The ZhouActTime function creates an enveloppe of tail currents from which the steady-state peak values and
    tail current values can be calculated which are needed for curve fitting. The voltage clamp protocols from Zhou
    et al. 1998 are used to calculate the time constants.
    
    Parameters
    ----------
    modeltype : Integer
        Determine the modeltype (See MMT file).
        
    temp : Integer
         Temperature input in degrees Celsius.
         
    sim : Myokit simulation protocol
         Simulation protocol.
    
    time : Integer
        Maximum time of all events in the protocol.
        
    tot_dur : Integer
        Total duration of one iteration.
            
    hold : Integer
        Duration of the holding potential.
             
    t_steps: Myokit simulation protocol
        Simulation protocol.
        
    showit : Integer
        Visualize a plot with score one and score two visualizes the curve fitting.
        
    showcurve : Integer
        Visualizes the curve fitting procedure for score one.
               
    log : Integer
        Create a logarithmic x-axis for visualization.
        

    Returns
    -------
    Dictionary with the peak, tail and time constant values
    """ 
    # Reset the initial states for each iteration 
    sim.reset()
    
    # Set the model type
    sim.set_constant('ikr.IKr_modeltype', modeltype) 
    
    # Set the maximum stepsize to 2ms
    sim.set_max_step_size(2) 
    
    # Set the temperature
    sim.set_constant('ikr_MM.IKr_temp', temp)
    
    # Set the extracellular potassium concentration 
    sim.set_constant('extra.Ko', 4)
    
    # Set tolerance to counter suboptimal optimalisation with CVODE
    sim.set_tolerance(1e-8, 1e-8)
    
    # Run the simulation protocol and log several variables
    dur = sim.run(time, log=['engine.time', 'membrane.V', 'ikr.IKr'])
    
    # Plot the curves for each step
    if showit == 1: 
        plt.subplot(2,1,1)
        plt.plot(dur['engine.time'], dur['membrane.V'])
        plt.xlabel('Time [ms]')
        plt.ylabel('Membrane potential [mV]')
        plt.title('Voltage-clamp protocol in steps')
        plt.subplot(2,1,2)
        plt.plot(dur['engine.time'], dur['ikr.IKr'])
        plt.title('Step-wise recorded traces')
        plt.xlabel('Time [ms]')
        plt.ylabel('pA/pF')
        plt.tight_layout()
        
    # Split the log into smaller chunks to overlay; to get the individual steps 
    ds = dur.split_periodic(tot_dur, adjust=True) 
    
    # Initialize the peak current variable
    Ikr_steady = np.zeros(len(ds)) 
    # Initialize the tail current variable
    Ikr_tail = np.zeros(len(ds)) 
    # Initialize the peak current variable
    Ikr_peak = np.zeros(len(ds))

    # Trim each new log to contain the steps of interest by enumerate through 
    # the individual duration steps     
    for k, d in enumerate(ds): 
        # Adjust is the time at the start of every sweep which is set to zero
        steady = d.trim_left(hold, adjust = True) 
        # Duration of the peak/steady current 
        steady = steady.trim_right(t_steps[k])
        # Total step duration (holding potential + peak current + margin of 1ms) to ensure the tail current
        tail = d.trim_left((hold + t_steps[k] + 1), adjust = True) 
        # Obtain the absolute tail current to prevent sign changes around reversal potential
        tail_IKr_abs = abs(np.array(tail['ikr.IKr'])) 
        # IKr tail amplitude was defined as current at the start of the repol. step minus the current at the end of repol.
        Ikr_tail[k] = max(tail_IKr_abs) - min(tail_IKr_abs)
        # Note, you could also just take the max of the tail which does not meaningfully affect the results (if any, few 100ths).
        #Ikr_tail[k] = max(tail_IKr_abs)
        # Calculate the peak steady current for each step
        Ikr_peak[k] = max(steady['ikr.IKr'])
        
        # Plot the voltage-clamp protocol together with the corresponding traces 
        if showit == 1:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(d['engine.time'], d['membrane.V'])
            plt.ylabel('Membrane potential [mV]')
            plt.title('Voltage-clamp protocol')
            plt.subplot(2,1,2)
            plt.plot(d['engine.time'], d['ikr.IKr'])
            plt.title('Step-wise recorded traces')
            plt.ylabel('Current density [pA/pF]')
            plt.xlabel('Time [ms]')
            plt.tight_layout()
     
    # Normalize the tail currents
    tail_norm = Ikr_tail/max(Ikr_tail)
     
    # Determine the time corresponding to the V1/2
    hlf = np.interp(0.5, tail_norm, t_steps)
     
    # Plot the normalized tail values
    if showit == 1:
        plt.figure()
        if log == 1:
            plt.semilogx(t_steps, tail_norm, 'o-', color = 'black', label = f'{temp}°C')
        else: 
            plt.plot(t_steps, tail_norm, 'o-', color = 'black', label = f'{temp}°C')
        plt.xlabel('Prepulse duration [ms]')
        plt.ylabel('Normalized tail current')
        plt.legend(title = 'Temperature')
        plt.suptitle('Normalized tail values')
        plt.tight_layout()
         
    # Initialize estimated parameters for curve fitting
    p0_act = (-1, 700, 1)
    
    # Fit the curve fit funcion with a mono-exponential approach
    par_opt, par_cov = curve_fit(mono_exp, t_steps, tail_norm, p0 = p0_act, maxfev = 3000)
    
    # Fitted time constants
    tau = par_opt[1]
    
    # Plot the curve fitting results together with the normalized tail current for each activation duration
    # step 
    if showcurve == 1: 
        plt.figure()
        if log == 0:
            plt.plot(t_steps, tail_norm,'o-', color = 'black', label = f'{temp}°C')
            plt.plot(t_steps, np.asarray(mono_exp(t_steps, -1, tau, 1)), '-', color = 'red', label = "Fitted Tau" )
        else:
            plt.semilogx(t_steps, tail_norm,'o-', color = 'black', label = f'{temp}°C')
            plt.semilogx(t_steps, np.asarray(mono_exp(t_steps, -1, tau, 1)), '-', color = 'red', label = "Fitted Tau" )
        plt.xlabel('Prepulse duration [ms]')
        plt.ylabel('Normalized tail current')
        plt.suptitle('Normalized tail values')
        plt.legend()
        plt.tight_layout()
        
    return dict(peak_val = Ikr_peak, tail_val = Ikr_tail, half_val = hlf, tail_norm = tail_norm, tau = tau)
#%% Calculate the time constants related to deactivation
def ZhouDeactTime(modeltype, temp, sim, time, tot_dur, showit = 0, showcurve = 0, log = 0):
    """ ZhouDeactTime function
    The ZhouDeactTime function can be used to calculate the deactivation time constants based on protocols
    from Zhou et al. 1998.
    
    Parameters
    ----------
    modeltype : Integer
        Determine the modeltype (See MMT file).
        
    temp : Integer
         Temperature input in degrees Celsius.
         
    sim : Myokit simulation protocol
         Simulation protocol.
    
    time : Integer
        Maximum time of all events in the protocol.
        
    tot_dur : Integer
        Total duration of one iteration.
        
    showit : Integer
        Visualize a plot with score one and score two visualizes the curve fitting.
        
    showcurve : Integer
        Visualizes the curve fitting procedure for score one.
               
    log : Integer
        Create a logarithmic x-axis for visualization.
          
    Returns
    -------
    Dictionary with the tail current duration, values, relative amplitudes and time constants.
    """ 
    # Reset the initial states for each iteration 
    sim.reset()
    
    # Set the model type
    sim.set_constant('ikr.IKr_modeltype', modeltype) 
    
    # Set the maximum stepsize to 2ms
    sim.set_max_step_size(2) 
    
    # Set the temperature
    sim.set_constant('ikr_MM.IKr_temp', temp)
    
    # Set the extracellular potassium concentration 
    sim.set_constant('extra.Ko', 4)
    
    # Set tolerance to counter suboptimal optimalisation with CVODE
    sim.set_tolerance(1e-8, 1e-8)
    
    # Run the simulation protocol and log several variables
    dur = sim.run(time, log=['engine.time', 'membrane.V', 'ikr.IKr'])
    
    # Plot the curves for each step
    if showit == 1: 
        plt.subplot(2,1,1)
        plt.plot(dur['engine.time'], dur['membrane.V'])
        plt.xlabel('Time [ms]')
        plt.ylabel('Membrane potential [mV]')
        plt.title('Voltage-clamp protocol in steps')
        plt.subplot(2,1,2)
        plt.plot(dur['engine.time'], dur['ikr.IKr'])
        plt.title('Step-wise recorded traces')
        plt.xlabel('Time [ms]')
        plt.ylabel('pA/pF')
        plt.tight_layout()
        
    # Split the log into smaller chunks to overlay; to get the individual steps 
    ds = dur.split_periodic(tot_dur, adjust=True) 
    
    # Initialize the peak current variable
    Ikr_steady = np.zeros(len(ds)) 
    # Initialize the tail current variable
    Ikr_tail = np.zeros(len(ds)) 
    # Initialize the fast time constants variable
    tau_f = np.zeros(len(ds))
    # Initialize the slow time constants variable
    tau_s = np.zeros(len(ds))
    # Initialize the relative amplitude variable
    rel_amp = np.zeros(len(ds))
    # Initialize the weighted time constant variable
    tau_w = np.zeros(len(ds))
        
    # Trim each new log to contain the steps of interest by enumerate through 
    # the individual voltage steps 
    for k, d in enumerate(ds):
        # Adjust is the time at the start of every sweep which is set to zero
        steady = d.trim_left(501, adjust = True) 
        # Duration of the peak/steady current 
        steady = steady.trim_right(1000) 
        # Total step duration (holding potential + peak current + margin of some ms) to ensure the tail current
        tail = d.trim_left(1505, adjust = True) 
        # Duration of the tail current (1000 ms)
        tail = tail.trim_right(1000) 
        # Obtain the absolute tail current to prevent sign changes around rev. potential
        tail_IKr_abs = abs(np.array(tail['ikr.IKr']))
        # Create an array that contains the duration of the tail currents
        tail_dur_deact = np.asarray(tail['engine.time'])
        
        # Initialize estimated parameters for curve fitting
        # parameter 'a1' = Zhou et al. 1998 shows a relative amplitude (at -70 mV) of (afast/(afast + aslow) ~ 0.8 of the actual amplitude
        # parameter 'a2' = Zhou et al. 1998 shows a relative amplitude (at -70 mV) of 1-(afast_/afast + aslow) ~ 0.2 of the actual amplitude
        # parameter 'tau1' = Derived from Zhou et al. 1998 experimental tau_fast is 64 ms 
        # parameter 'tau2' = Derived from experimental Zhou et al. 1998 tau_slow is 303 ms
        # parameter 'c' = Deactivation shows decay, so always zero
        # Note, tau always needs to be positive
        a1 = 0.8
        tau1 = 64
        a2 = 0.2
        tau2 = 303
        con = 0
        p0 = (a1, tau1, a2, tau2, con)
        
        # Perform curve fitting w/ double exponential function for deactivation to obtain fast and slow tau
        # Note, set bounds otherwise the curve fitting will overflow. The lower bounds are always zero, 
        # because the tau cannot be negative. In this case, a pulse of 1 ms was used and thus the max for 
        # tau is roughly 500 ms.
        par_opt, par_cov = curve_fit(double_exp, tail_dur_deact, tail_IKr_abs, p0 = p0, maxfev = 3000, bounds = (0, [10, 500, 10, 500, 1]))
        
        # Fitted time constants (tau fast and tau slow)
        # Note, there is a potential flipping of tau_f and tau_s during optimization
        # (e.g. first: tau1 < tau2, after optimization tau1 > tau2)
        # Create an if-statement to counter this problem. Furthermore, the relative amplitude 
        # is (afast/(afast + aslow))
        if par_opt[1] < par_opt[3]:
            tau_f[k] = par_opt[1]
            tau_s[k] = par_opt[3]
            rel_amp[k] = par_opt[0]/(par_opt[0] + par_opt[2])
        else:
            tau_f[k] = par_opt[3]
            tau_s[k] = par_opt[1]
            rel_amp[k] = par_opt[2]/(par_opt[0] + par_opt[2])

        # Calculate the tau weighted by multiplying the relative amplitude with the fast time constant
        # and the 1-relative amplitude with the slow time constant. 
        tau_w = (rel_amp * tau_f) + ((1-rel_amp) * tau_s)
    
        # Plot the curve fitting procedure
        if showcurve == 1:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(tail['engine.time'], tail['membrane.V'])
            plt.ylabel('Membrane potential [mV]')
            plt.xlabel('Time [ms]')
            plt.subplot(2,1,2)
            plt.plot(tail_dur_deact, tail_IKr_abs, '.',tail_dur_deact, np.asarray(double_exp(tail_dur_deact, *par_opt)), '-')
            plt.ylabel('Current [pA]')
            plt.xlabel('Time [ms]')
            plt.tight_layout()
        
        # Plot the voltage-clamp protocol together with the corresponding traces 
        if showit == 1:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(d['engine.time'], d['membrane.V'])
            plt.ylabel('Membrane potential [mV]')
            plt.title('Voltage-clamp protocol')
            plt.subplot(2,1,2)
            plt.plot(d['engine.time'], d['ikr.IKr'])
            plt.title('Step-wise recorded traces')
            plt.ylabel('Current density [pA/pF]')
            plt.xlabel('Time [ms]')
            plt.tight_layout()
    
    return dict(tail_dur = tail_dur_deact, tail_abs = tail_IKr_abs, tail_val = Ikr_tail, tau_fast = tau_f, tau_slow = tau_s, rel_amp = rel_amp, tau_weight = tau_w)
#%% Create a function to calculate the shift in midpoint activation between temperatures
def ZhouActVolt(modeltype, temp, sim, time, tot_dur, v_steps, showit = 0):
    """ ZhouActVolt function
    The ZhouActVolt function creates a voltage clamp protocol that allows to compare the temperature-dependent 
    shift in half maximal channel activation between two temperatures. This is based on protocols from Zhou et al.
    1998.
    
    Parameters
    ----------
    modeltype : Integer
        Determine the modeltype (See MMT file).
        
    temp : Integer
         Temperature input in degrees Celsius.
         
    sim : Myokit simulation protocol
         Simulation protocol.
    
    time : Integer
        Maximum time of all events in the protocol.
        
    tot_dur : Integer
        Total duration of one iteration.
        
    v_steps : List
        Voltage steps.
        
    showit : Integer
        Visualize a plot with score one and score two visualizes the curve fitting.
          
    Returns
    -------
    Dictionary with peak, tail and V1/2 values.
    """ 

    # Reset the initial states for each iteration 
    sim.reset()
    
    # Set the model type
    sim.set_constant('ikr.IKr_modeltype', modeltype) 
    
    # Set the maximum stepsize to 2ms
    sim.set_max_step_size(2) 
    
    # Set the temperature
    sim.set_constant('ikr_MM.IKr_temp', temp)
    
    # Set the extracellular potassium concentration 
    sim.set_constant('extra.Ko', 4)
    
    # Set tolerance to counter suboptimal optimalisation with CVODE
    sim.set_tolerance(1e-8, 1e-8)
    
    # Run the simulation protocol and log several variables
    dur = sim.run(time, log=['engine.time', 'membrane.V', 'ikr.IKr'])
    
    # Plot the curves for each step
    if showit == 1: 
        plt.subplot(2,1,1)
        plt.plot(dur['engine.time'], dur['membrane.V'])
        plt.xlabel('Time [ms]')
        plt.ylabel('Membrane potential [mV]')
        plt.title('Voltage-clamp protocol in steps')
        plt.subplot(2,1,2)
        plt.plot(dur['engine.time'], dur['ikr.IKr'])
        plt.title('Step-wise recorded traces')
        plt.xlabel('Time [ms]')
        plt.ylabel('pA/pF')
        plt.tight_layout()
        
    # Split the log into smaller chunks to overlay; to get the individual steps 
    ds = dur.split_periodic(tot_dur, adjust=True) 
    
    # Initialize the peak current variable
    Ikr_steady = np.zeros(len(ds)) 
    # Initialize the tail current variable
    Ikr_tail = np.zeros(len(ds)) 
    
    # Trim each new log to contain the steps of interest by enumerate through 
    # the individual voltage steps 
    for k, d in enumerate(ds):
        # Adjust is the time at the start of every sweep which is set to zero
        steady = d.trim_left(500, adjust = True) 
        # Duration of the peak/steady current 
        steady = steady.trim_right(4000) 
        # Total step duration (holding potential + peak current + margin of some ms) to ensure the tail current
        tail = d.trim_left(4501, adjust = True) 
        # Duration of the tail current (5000 ms)
        tail = tail.trim_right(5000) 
        # Obtain the absolute tail current to prevent sign changes around rev. potential
        tail_IKr_abs = abs(np.array(tail['ikr.IKr']))
        # IKr tail amplitude was defined as current at the start of the repol. step minus the current at the end of repol.
        Ikr_tail[k] = max(tail['ikr.IKr']) - min(tail['ikr.IKr'])  
        # Note, you could also just take the max of the tail which does not meaningfully affect the results (if any, few 100ths).
        #Ikr_tail[k] = max(tail_IKr_abs)
        
        # Plot the voltage-clamp protocol together with the corresponding traces 
        if showit == 1:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(d['engine.time'], d['membrane.V'])
            plt.ylabel('Membrane potential [mV]')
            plt.title('Voltage-clamp protocol')
            plt.subplot(2,1,2)
            plt.plot(d['engine.time'], d['ikr.IKr'])
            plt.title('Step-wise recorded traces')
            plt.ylabel('Current density [pA/pF]')
            plt.xlabel('Time [ms]')
            plt.tight_layout()

    # Normalize the tail current
    tail_norm = Ikr_tail/max(Ikr_tail)
    
    # Determine the voltage corresponding to the V1/2 current
    hlf = np.interp(0.5, tail_norm, v_steps)
    
    return dict(peak_val = Ikr_steady, tail_abs = tail_IKr_abs, tail_val = Ikr_tail, tail_norm = tail_norm, half_val = hlf)
#%% Create a function to calculate the time constants related to inactivation

def ZhouInactTime(modeltype, temp, sim, time, tot_dur, showit = 0, showcurve = 0):
    """ ZhouInactTime function
    The ZhouInactTime function can be used to calculate the inactivation time constants based on protocols
    from Zhou et al. 1998.
    
     
    Parameters
    ----------
    modeltype : Integer
        Determine the modeltype (See MMT file).
        
    temp : Integer
         Temperature input in degrees Celsius.
         
    sim : Myokit simulation protocol
         Simulation protocol.
    
    time : Integer
        Maximum time of all events in the protocol.
        
    tot_dur : Integer
        Total duration of one iteration.
        
    showit : Integer
        Visualize a plot with score one and score two visualizes the curve fitting.
        
    showcurve : Integer
        Visualizes the curve fitting procedure for score one.
               
          
    Returns
    -------
    Dictionary that contains tail current values, duration and time constants.
    """ 
    # Reset the initial states for each iteration 
    sim.reset()
    
    # Set the model type
    sim.set_constant('ikr.IKr_modeltype', modeltype) 
    
    # Set the maximum stepsize to 2ms
    sim.set_max_step_size(2) 
    
    # Set the temperature
    sim.set_constant('ikr_MM.IKr_temp', temp)

    # Set the extracellular potassium concentration
    sim.set_constant('extra.Ko', 4)
    
    # Set tolerance to counter suboptimal optimalisation with CVODE
    sim.set_tolerance(1e-8, 1e-8)
    
    # Run the simulation protocol and log several variables
    dur = sim.run(time, log=['engine.time', 'membrane.V', 'ikr.IKr'])
    
    # Plot the curves for each step
    if showit == 1: 
        plt.subplot(2,1,1)
        plt.plot(dur['engine.time'], dur['membrane.V'])
        plt.xlabel('Time [ms]')
        plt.ylabel('Membrane potential [mV]')
        plt.title('Voltage-clamp protocol in steps')
        plt.subplot(2,1,2)
        plt.plot(dur['engine.time'], dur['ikr.IKr'])
        plt.title('Step-wise recorded traces')
        plt.xlabel('Time [ms]')
        plt.ylabel('pA/pF')
        plt.tight_layout()
        
    # Split the log into smaller chunks to overlay; to get the individual steps 
    ds = dur.split_periodic(tot_dur, adjust=True) 
    
    # Initialize the tail current variable
    Ikr_tail = np.zeros(len(ds)) 
    # Initialize the time constant variable 
    tau = np.zeros(len(ds))

    # Trim each new log to contain the steps of interest by enumerate through 
    # the individual voltage steps 
    for k, d in enumerate(ds):
        # Total step duration (holding potential + peak current + repolarizing current) to ensure the tail current
        tail = d.trim_left(702, adjust = True) 
        # Duration of the tail current (20 ms)
        tail = tail.trim_right(20) 
        # Obtain the absolute tail current to prevent sign changes around rev. potential
        tail_IKr_abs = abs(np.array(tail['ikr.IKr'])) 
        # Create an array that contains the duration of the tail currents
        tail_dur_inact = np.asarray(tail['engine.time'])
        
        # Initialize estimated parameters for curve fitting
        a = -1
        t = 10
        c = 0
        p0 = (a, t, c)
        
        # Fit the curve fit function with a mono exponential approach
        par_opt, par_cov = curve_fit(mono_exp, tail_dur_inact, tail_IKr_abs, p0 = p0, maxfev = 3000)
    
        # Fitted time constants
        tau[k] = par_opt[1]
    
        # Plot the curve fitting procedure
        if showcurve == 1:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(tail['engine.time'], tail['membrane.V'])
            plt.ylabel('Membrane potential [mV]')
            plt.xlabel('Time [ms]')
            plt.subplot(2,1,2)
            plt.plot(tail_dur_inact, tail_IKr_abs, '.',tail_dur_inact, np.asarray(mono_exp(tail_dur_inact, *par_opt)), '-')
            plt.ylabel('Current [pA]')
            plt.xlabel('Time [ms]')
            plt.tight_layout()
        
        # Plot the voltage-clamp protocol together with the corresponding traces 
        if showit == 1:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(d['engine.time'], d['membrane.V'])
            plt.ylabel('Membrane potential [mV]')
            plt.title('Voltage-clamp protocol')
            plt.subplot(2,1,2)
            plt.plot(d['engine.time'], d['ikr.IKr'])
            plt.title('Step-wise recorded traces')
            plt.ylabel('Current density [pA/pF]')
            plt.xlabel('Time [ms]')
            plt.tight_layout()
        
    return dict(tail_dur = tail_dur_inact, tail_abs = tail_IKr_abs, tail_val = Ikr_tail, tau = tau)  

#%% Create a function that allows to calculate the time constants related to recovery from inactivation 
def ZhouRecovTime(modeltype, temp, sim, time, tot_dur, showit = 0, showcurve = 0):
    """ ZhouRecovTime function
    The ZhouRecovTime function can be used to calculate the recovery from inactivation time constants.
    Note, the time constants are calculated with both a mono exponent (> -40 mV) and a double exponent (<= - 40 mV)
    similar to Zhou et al. 1998. 
    
    Parameters
    ----------
    modeltype : Integer
        Determine the modeltype (See MMT file).
        
    temp : Integer
         Temperature input in degrees Celsius.
         
    sim : Myokit simulation protocol
         Simulation protocol.
    
    time : Integer
        Maximum time of all events in the protocol.
        
    tot_dur : Integer
        Total duration of one iteration.
        
    showit : Integer
        Visualize a plot with score one and score two visualizes the curve fitting.
        
    showcurve : Integer
        Visualizes the curve fitting procedure for score one.
               
                  
    Returns
    -------
    Dictionary that contains the tail current value, duration and time constants (mono, fast and slow).
    """ 
    # Reset the initial states for each iteration 
    sim.reset()
    
    # Set the model type
    sim.set_constant('ikr.IKr_modeltype', modeltype) 
    
    # Set the maximum stepsize to 2ms
    sim.set_max_step_size(2) 
    
    # Set the temperature
    sim.set_constant('ikr_MM.IKr_temp', temp)
        
    # Set the extracellular potassium concentration
    sim.set_constant('extra.Ko', 4)
    
    # Set tolerance to counter suboptimal optimalisation with CVODE
    sim.set_tolerance(1e-8, 1e-8)
    
    # Run the simulation protocol and log several variables
    dur = sim.run(time, log=['engine.time', 'membrane.V', 'ikr.IKr'])
    
    # Plot the curves for each step
    if showit == 1: 
        plt.subplot(2,1,1)
        plt.plot(dur['engine.time'], dur['membrane.V'])
        plt.xlabel('Time [ms]')
        plt.ylabel('Membrane potential [mV]')
        plt.title('Voltage-clamp protocol in steps')
        plt.subplot(2,1,2)
        plt.plot(dur['engine.time'], dur['ikr.IKr'])
        plt.title('Step-wise recorded traces')
        plt.xlabel('Time [ms]')
        plt.ylabel('pA/pF')
        plt.tight_layout()
        
    # Split the log into smaller chunks to overlay; to get the individual steps 
    ds = dur.split_periodic(tot_dur, adjust=True) 
    
    # Initialize the tail current variable
    Ikr_tail = np.zeros(len(ds)) 
    # Initialize the mono exponential time constant
    tau_m = np.zeros(len(ds))
    # Initialize the fast time constants variable
    tau_f = np.zeros(len(ds))
    # Initialize the slow time constants variable
    tau_s = np.zeros(len(ds))
    # Initialize the afast variable
    a_fast = np.zeros(len(ds))
    # Initialize the aslow variable
    a_slow = np.zeros(len(ds))
    
    # Trim each new log to contain the steps of interest by enumerate through 
    # the individual voltage steps 
    for k, d in enumerate(ds):
        # Trim the tail current
        tail = d.trim_left(700, adjust = True) 
        # Duration of the tail current (10ms or 20ms)
        tail = tail.trim_right(20) 
        # Obtain the absolute tail current to prevent sign changes around rev. potential
        tail_IKr_abs = abs(np.array(tail['ikr.IKr']))
        # Create an array that contains the duration of the tail currents
        tail_dur_recov = np.asarray(tail['engine.time'])
        
        # In Zhou et al. 1998 pg. 235, they write: 'The time constant of recovery from inactivation
        # was measured as the mono exponential fit to the tail current rising phase (> -40 mV) or as the fast
        # time constant of a double exponential fit (<= -40 mV), where deactivation is present in the tail current
        # deactivation is slow
        
        # Initialize p0 for both a mono exponential function and double exponential function
        p0_mono = (-1, 5, 1) 
        p0_bi = (0.8, 3, 0.2, 30, 1)
        
        if k < 1:
            # Fit the curve fit function with a mono exponential approach
            par_opt, par_cov = curve_fit(mono_exp, tail_dur_recov, tail_IKr_abs, p0 = p0_mono, maxfev = 3000)
        
            # Fitted time constants
            tau_m[k] = par_opt[1]
      
            # Plot the curve fitting procedure
            # Note, the monoexponent levels off after the increase and does not
            # decrease like the biexponent. This is also the reason why at <-40 mV
            # the curve fitting is performed with a double exponent, because after
            # the initial recovery there is deactivation at these low voltages.
            if showcurve == 1:
                plt.figure()
                plt.subplot(2,1,1)
                plt.plot(tail['engine.time'], tail['membrane.V'])
                plt.ylabel('Membrane potential [mV]')
                plt.xlabel('Time [ms]')
                plt.subplot(2,1,2)
                plt.plot(tail_dur_recov, tail_IKr_abs, '.',tail_dur_recov, np.asarray(mono_exp(tail_dur_recov, *par_opt)), '-')
                plt.ylabel('Current [pA]')
                plt.xlabel('Time [ms]')
                plt.tight_layout()
        else: 
            # Fit the curve fit function with a double exponential approach
            # Note, put bounds of zero on taus, because they cannot be negative.
            # The a's can be negative due to the initial increase in current 
            # (recovery from inactivation) followed by a decrease in current
            # due to deactivation. Therefore, the a's move in opposite directions
            # where afast represents the recovery and aslow the deactivation. This
            # is only true for a biexponent.
            par_opt, par_cov = curve_fit(double_exp, tail_dur_recov, tail_IKr_abs, p0 = p0_bi, maxfev = 3000, bounds = ([-10, 0, -10, 0, -10], [10, 200, 10, 200, 10]))
            
            # Fitted time constants (fast and slow)
            if par_opt[1] < par_opt[3]:
                tau_f[k] = par_opt[1]
                tau_s[k] = par_opt[3]
                a_fast[k] = par_opt[0]
                a_slow[k] = par_opt[2]
            else:
                tau_f[k] = par_opt[3]
                tau_s[k] = par_opt[1]
                a_fast[k] = par_opt[2]
                a_slow[k] = par_opt[0]
        
            # Plot the curve fitting procedure
            if showcurve == 1:
                plt.figure()
                plt.subplot(2,1,1)
                plt.plot(tail['engine.time'], tail['membrane.V'])
                plt.ylabel('Membrane potential [mV]')
                plt.xlabel('Time [ms]')
                plt.subplot(2,1,2)
                plt.plot(tail_dur_recov, tail_IKr_abs, '.',tail_dur_recov, np.asarray(double_exp(tail_dur_recov, *par_opt)), '-')
                plt.ylabel('Current [pA]')
                plt.xlabel('Time [ms]')
                plt.tight_layout()
        
        # Plot the voltage-clamp protocol together with the corresponding traces 
        if showit == 1:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(d['engine.time'], d['membrane.V'])
            plt.ylabel('Membrane potential [mV]')
            plt.title('Voltage-clamp protocol')
            plt.subplot(2,1,2)
            plt.plot(d['engine.time'], d['ikr.IKr'])
            plt.title('Step-wise recorded traces')
            plt.ylabel('Current density [pA/pF]')
            plt.xlabel('Time [ms]')
            plt.tight_layout()
              
    return dict(tail_dur = tail_dur_recov, tail_abs = tail_IKr_abs, tail_val = Ikr_tail, tau_fast = tau_f, tau_slow = tau_s, tau_mono = tau_m, afast = a_fast, aslow = a_slow)
