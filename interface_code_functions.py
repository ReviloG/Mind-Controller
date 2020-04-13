################################################################################
#
#                               Libraries
#
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
from matplotlib.pyplot import show
import pandas as pd

################################################################################
#
#                               Functions
#
################################################################################

def name(**x):
    return x

"""read some number of values (N)
    read a sample of (N) values on 8 channels"""


"""see what bands they fit in
    compare the 8-channel sample to each ML learned bands
        Does the 8-channel sample fit into this band or band combination?
        (Band combination could be disjoint bands or some kind of linear(?) combination)
        [Will need to identify what bands are important for each signal,
         and how to handle the unimportant bands.  Do we just ignore them?]"""

def generate_samples(args):   
    """
    Function that generates randomized amplitude,
    phase, and frequency sine data.
    """
    samples = np.random.normal(args['mean'],args['std'], (8, args['sample_size']))
    for i in range(args['n_channels']):
        A = np.random.randint(1, 17)
        w = np.random.randint(1,17)
        theta = (np.random.randint(1,181))/(np.pi)
        sine = A * np.sin(w * np.linspace(0, 2 * np.pi, args['sample_size']) - \
                      theta)
        samples[i] = sine + samples[i]
    return samples

       
def plot_samples(args, samples):
    """
    Fuction that plots all the sample data
    """
    for i in range(args['n_channels']):
        plt.subplot(args['n_channels'], 1, i+1)
        plt.plot(samples[i])
    plt.show()



def upper_band(data, N, D):
    running_mean(data, N)
    return MA + D * np.std(data - MA)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_bollinger(args, signal, my_list):
    for i,v in enumerate(my_list):
        rm = running_mean(signal, v)
        plt.subplot(len(my_list),1,i+1)
        plt.plot(signal, label = 'original')
        plt.plot(rm, label = '{}'.format(v))
        plt.legend()
    plt.show()

def plot_bollinger_all(args, signal, my_list):
    plt.plot(signal, label = 'original')
    for i,v in enumerate(my_list):
        rm = running_mean(signal, v)
        plt.plot(rm, label = '{}'.format(v))
    plt.legend()
    plt.show()


def plot_Bollinger_function(data, w_size, n_stds):
    m, u, l = Bollinger_Bands(data, w_size, n_stds)
    plt.plot(m, label ='mean')
    plt.plot(l, label = 'lower')
    plt.plot(u, label = 'upper')
    plt.legend()
    plt.show()



def Bollinger_Bands(data, w_size, n_stds):

    rolling_mean = data.rolling(window=w_size).mean()
    rolling_std  = data.rolling(window=w_size).std()
    upper_band = rolling_mean + (rolling_std*n_stds)
    lower_band = rolling_mean - (rolling_std*n_stds)

    return rolling_mean, upper_band, lower_band

def plot_bollinger_bands(data, args, w_size, n_stds):
    x, z, y = Bollinger_Bands(data, w_size, n_stds)
    for i in range(args['n_channels']):
        plt.subplot(args['n_channels'], 1, i + 1)
        plt.plot(data[i], label = 'data')
        plt.plot(x[i])
        plt.plot(y[i])
        plt.plot(z[i])
        plt.legend()
    plt.show()

def to_dataframe(args, samples):
    #Put the data in the appropriate form for the Bollinger Bands plot.
    return pd.DataFrame(samples).T





"""use that band
    if a channel is identified, send the info for that channel to the program
        (info discrete or cont(the UP channel, the DOWN channel is identified))
        [Labels what band is being identified, and outputs the
         appropriate command to the program. After each data reading, the signal
         is read again and commanded again.  The slider will remain where it
         is and each reading iteration  will move it in the
         appropriate direction]"""


def plot_single_Bollinger(signal, w_size, n_stds):
    #Signal in the form of a DataFrame where the time index is a column.
    rm, ub, lb = Bollinger_Bands(signal, w_size, n_stds)
    plot(signal, label = 'signal')
    plot(rm, label = 'mean')
    plot(ub, label = 'upper')
    plot(lb, label = 'lower')
    plt.legend()
    show()
    return rm, ub, lb

def compare_signal_and_band(signal, bands):
    """"Returns the error % outside of the bounds,
    1 minus the output percentage is the % correct match"""
    ub = bands['ub']
    lb = bands['lb']
    #Signal is pandas dataframe w/ column a time series
    #Band is dictionary with 'm' median 'ub' upper bound 'lb' lower bound time
    #series

    #Count how many times it's outside of the band to calculate percentage
    outside_count = np.sum((signal < lb) | (signal > ub))[0]
    error = outside_count / len(signal)
    return error
    


def plot_signal_with_bands(signal, bands, disturbance):
    """Plot a signal alongside bands created with another signal"""
    plt.plot(signal + disturbance, label = 'signal')
    ub = bands['ub']
    lb = bands['lb']
##    for name in bands.keys():
##        plt.plot(bands[name], label = name)
    plt.plot(ub, label = 'ub')
    plt.plot(lb, label = 'lb')
    plt.legend()

    error = compare_signal_and_band(signal + disturbance, bands)
    #1 - error is the percentage match
    plt.title('{} match with disturbance {}'.format(np.round(1-error, 3), disturbance))
    plt.show()



def plot_errors(signal, bands):
    #Plot the errors of the signal in two parts:  the errors for the
    # signal above the upper band and the errors for the signal
    #below the lower band.
    
    ub = bands['ub']
    lb = bands['lb']

    upper_error = signal - ub
    lower_error = lb - signal
    upper_mask = upper_error > 0
    lower_mask = lower_error > 0

    upper_error[(signal - ub) <= 0] = 0
    lower_error[(lb - signal) <= 0] = 0

    plot(upper_error, label = 'upper_error')
    plot(lower_error, label = 'lower_error')

    plt.legend()
    plt.show()
    return upper_error, lower_error


def generate_samples_2(args):   
    """
    Function that generates randomized amplitude,
    phase, and frequency sine data.
    """
    samples = np.random.normal(args['mean'],args['std'], (8, args['sample_size']))
    for i in range(args['n_channels']):
        A = np.random.randint(1,4)
        w = np.random.randint(1,4)
        theta = (np.random.choice([0, 45, 90]))/(np.pi)
        sine = A * np.sin(w * np.linspace(0, 2 * np.pi, args['sample_size']) - \
                      theta)
        samples[i] = sine + samples[i]
    return samples

#Get the Bollinger Bands
def generate_n_bollinger_bands(args, n, w_size, n_stds):
    """
    Generate three dictionaries, each containing
    the median, upper, and lower bands for each action in the program
    that Stephen is writing.
    """

    #Us is dictionary of DataFrames of the shape m x n where m is the samples
    # and n is the 8 channels.  Ms[0], Us[0], Ls[0] is the 1st Bollinger bands
    #and etc.
    Ms, Us, Ls = {}, {}, {}
    for i in range(n):
        band_stream = generate_samples_2(args)
        band_stream = to_dataframe(args, band_stream)
        Ms[i], Us[i], Ls[i] = Bollinger_Bands(band_stream, w_size, n_stds)

    bands = {'Ms':Ms, 'Us':Us, 'Ls': Ls}
    return bands

def match_test(signal, lb, ub):
    """
    Count the number of times the signal is outside of the bounds then
    divide by the length of the length of the signal.
    """
    outside_count = np.sum((signal < lb) | \
                                   (signal > ub))
    error = outside_count / len(signal)
    return 1-error

def get_total_error(stream, lb, ub):
    """
    Sum all the differences of values that overshoot or undershoot
    from the upper and lower bound, respectfully.
    """

    above = (stream > ub)
    below = (stream < lb)
    upper_error = stream[above] - ub[above]
    lower_error = lb[below] - stream[below]
    total_error = np.sum(upper_error) + np.sum(lower_error)
    return total_error.astype(int)





def check_good_region(correct_list, errors_list, args,
                      band_weights_list=None):

    if band_weights_list == None:
        band_weights_list = 3 *[(1/args['n_channels']) * \
                                np.ones(args['n_channels'])]

    """

    Check to see if the correct_list, and error_list values (cl[i], el[i]) are
    in the good regions.  Default weights are set to uniform.
    
    Takes in three lists of length 3, all of shape (n_channels,1)
    Checks to see if each data point from correct_list and error_list
    is in the defined good region.  A second array of ones will be created.
    If the data point "i" is in the good region, then the new array [i]
    will be changed to a 1.  This is repeated until the array is full of 1's.
    Then array will then be multiplied by the weight matrix and the sum
    of the array will be calculated.  This will be done with each channel
    and return the sum scores for each channel.
    
    """
    
    scores = []
    for i in range(len(correct_list)):
        good_percentage_count = (correct_list[i] >= 0.5)
        good_error_count = (errors_list[i] < 500)

        overall_good = np.array(1*(good_percentage_count & good_error_count))
        overall_score = np.sum(overall_good * band_weights_list[i])
        scores.append(overall_score)

    scores = np.array(scores)
  
    return scores


def compare_stream_to_bands(stream, bands, args):
    """
    Compare the stream to the bands by counting the number
    of data points outside of the bands.  Then check to see if the percentages
    correct are over the command threshold, and check to see whether or not
    more than half of the channels are matched to over the command threshold
    (50% in this case).  If both conditions are met, then the command with the
    highest match percentages is returned, otherwise the command "Do nothing"
    is returned.

    Parameters
    ----------
    

    stream : pandas dataframe
            n_channels x sample_size array of stream data
            
    bands : dictionary
            Dictionary containing the bands, where bands.keys are [Ms, Us, Ls]
            and each Ms is a dictionary with bands['Ms'].keys() = [0,1,2]
            with one for each median, upper bound, and lower bound.

    args : dictionary
            Contains all the constants.
            
            
    Returns
    -------

    command : string
            Command for Stephen to use for the program.
    
    """
    correct_list = []
    for i in range(len(bands)):
        ub = bands['Us'][i]
        lb = bands['Ls'][i]
        match = match_test(stream, lb, ub)
        correct_list.append(match)

    errors_list = []
    for i in range(len(bands)):
        ub = bands['Us'][i]
        lb = bands['Ls'][i]
        total_errors = get_total_error(stream, lb, ub)
        errors_list.append(total_errors)

    scores = check_good_region(correct_list, errors_list, args)
    max_score = np.max(scores)
    if max_score < 0.5:
        return 'Do nothing'
    
    command_index = np.argmax(scores)
    commands = ['U', 'D', 'B']
    return commands[command_index]


def plot_bands_and_stream(stream, bands, args):

    """

    Plot the three bands along with the stream channel data
    and display the match percentages and total errors.
    
    """
    names = ['Button', 'Up', 'Down']
    n_bands = len(bands)
    
    for i in range(n_bands):
        fig, ax = plt.subplots(args['n_channels'],1)
        for j in range(args['n_channels']):
            
            stream_channel_j = stream.iloc[:, j]
            ax[j].plot(stream_channel_j, c='green')
    
            ub = bands['Us'][i].iloc[:, j]
            lb = bands['Ls'][i].iloc[:, j]

            match = match_test(stream_channel_j, lb, ub)

            total_error = get_total_error(stream_channel_j, lb, ub)
            
            ax[j].plot(ub, c='blue')
            ax[j].plot(lb, c='red')

            #Set position and plot the error for the band.
            ax[j].set_yticks([0.5])
            ax[j].set_yticklabels(['({}, {})'.format(np.round(match, 2),
                                                   total_error)])
            
        fig.suptitle('{}'.format(names[i]))
        plt.show()


def plot_bands_and_stream_together(stream, bands, args):

    """

    Plot the three bands along with the stream channel data
    and display the match percentages and total errors in a SINGLE figure.
    
    """
    names = ['Button', 'Up', 'Down']
    n_bands = len(bands)
    
    fig, ax = plt.subplots(args['n_channels'],3)
    for i in range(len(bands)):
        for j in range(args['n_channels']):

                if j == 0:
                    ax[j,i].set_title('{}'.format(names[i]))
                stream_channel_j = stream.iloc[:, j]
                ax[j,i].plot(stream_channel_j, c='green', label = 'signal')
        
                ub = bands['Us'][i].iloc[:, j]
                lb = bands['Ls'][i].iloc[:, j]

                match = match_test(stream_channel_j, lb, ub)

                total_error = get_total_error(stream_channel_j, lb, ub)
                
                ax[j,i].plot(ub, c='blue', label='upper bound')
                ax[j,i].plot(lb, c='red', label='lower bound')

                #Set position and plot the error for the band.
                ax[j,i].set_yticks([0.5])
                ax[j,i].set_yticklabels(['({}, {})'.format(np.round(match, 2),
                                                        total_error)])

                ax[j,i].set_xticks([0.5])
                ax[j,i].set_xticklabels([''])
                
    plt.legend()
    plt.show()
