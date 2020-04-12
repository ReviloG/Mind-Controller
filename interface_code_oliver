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

##def get_sample(args):
## """Function that returns the live sample readings"""
##    sample_size = args['sample_size']
##    sample = np.zeros(sample_size)
##    return sample

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



################################################################################
#
#                               Tests
#
################################################################################

# Set up the data

#8 channels, 1000 images, 0 mean for the noise, standard deviation 1,
#command_threshold is the % that must be crossed to be considered correct.
args = name(n_channels = 8, sample_size = 1000, mean = 0, std = 1,
            command_threshold = 0.50 )



#(1.) Get sample sine waves and plot them
samples = generate_samples(args)
plot_samples(args, samples)


#(2.) Setting up the bands and plot the moving average
test = np.random.randn(args['sample_size'])
moving_average_list = [2*n+2 for n in range(10)]
plot_bollinger(args, samples[-1], moving_average_list)
plot_bollinger_all(args, samples[-1], moving_average_list)



#(3.) Get test data and plot the bollinger bands for the data
test = np.random.randn(args['sample_size'])
data = pd.DataFrame(test)

samples = generate_samples(args)
data2 = pd.DataFrame(samples).T

x, z, y = Bollinger_Bands(data2, w_size = 8, n_stds = 2)

plot_Bollinger_function(data, w_size = 8, n_stds = 2)



#(4.) Generate sample sine waves and plot their Bollinger bands with
      #window size 8 and std 2
samples = generate_samples(args)
data2 = to_dataframe(args, samples)
n_stds = 2
w_size = 8
plot_bollinger_bands(data2, args, w_size = w_size, n_stds = n_stds)

#(5.) Create list of DataFrames that represent signal data
      #Plot the 0th DataFrame
data = [to_dataframe(args, generate_samples(args)) for i in range(100)]
plot_bollinger_bands(data[0], args, w_size = 8, n_stds = 2)



#(6.)  #Create a signal, calculate all its bands, and plot it
       #with the bands.  Plot it with a disturbance.
signal = pd.DataFrame(np.random.randn(1,1000) + \
                           5 * np.sin(np.linspace(0, 10 * np.pi, 1000))).T
rm, ub, lb = plot_single_Bollinger(signal, w_size = 4, n_stds = 2)

#Create a dictionary of bands
bands= {'ub':ub, 'lb':lb, 'rm':rm}

#Calculate the error for the test code created three lines ago.
#Added +1 to the signal as a test to see how the error function would work.
disturbance = 1

error = compare_signal_and_band(signal + disturbance, bands)

plot_signal_with_bands(signal, bands, disturbance)

#(7.) Plot the errors to see where they occur the most.
noise = np.random.randn(1, 1000)
noise = pd.DataFrame(noise).T
upper_error, lower_error = plot_errors(signal + noise, bands)

#(8.) Simulate a stream and 3 bands and compare them to each band, plot the
      #comparisons with error percentages.

#Set window and number of standard deviations for the Bollinger Bands
w_size = 8
n_stds = 2



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

def get_total_error(signal, lb, ub):
    """
    Sum all the differences of values that overshoot or undershoot
    from the upper and lower bound, respectfully.
    """

    c = signal
    b = lb
    a = ub

    above = (c > a)
    below = (c < b)
    upper_error = c[above] - a[above]
    lower_error = b[below] - c[below]
    total_error = np.sum(upper_error) + np.sum(lower_error)
    return int(total_error)


#Create band data
bands = generate_n_bollinger_bands(args, 3, w_size, n_stds)


#Create the stream data
stream = generate_samples_2(args)
stream = to_dataframe(args, stream)

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

    stream : array_like
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
            Command for Stephen to use for his program.
    
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



    commands_list = []
    for i in range(len(correct_list)):
        command_threshold_check = (correct_list[i] > args['command_threshold'])
        more_than_half_check = (np.sum(command_threshold_check) >= 4)
        commands_list.append(int(more_than_half_check))

    if np.max(commands_list) == 0:
        return 'Do nothing'
    command_index = np.argmax(commands_list)
    commands = ['U', 'D', 'B']
    return commands[command_index]


command = compare_stream_to_bands(stream, bands, args)



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


def plot_bands_and_stream_together(streams, bands, args):

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

plot_bands_and_stream_together(stream, bands, args)
            
        

plot_bands_and_stream(stream, bands, args)

# (9.) Find a better similarity check between two signals.
def test_9(variance):
    base = np.sin(3 * np.linspace(0, 2* np.pi, 1000))
    a = 1 + base
    b = a - 2
    c =  base + np.random.normal(0,variance,1000)
    line = np.zeros((len(a), 1))

    plt.plot(a, label='upper', color='blue')
    plt.plot(b, label='lower', color = 'red')
    plt.plot(c, label = 'data', color='green')
    plt.plot(line, color='black')

    above = (c > a)
    below = (c < b)
    upper_error = c[above] - a[above]
    lower_error = b[below] - c[below]
    total_error = np.sum(upper_error) + np.sum(lower_error)


    outside = (c > a) | (c < b)
    error = np.sum(outside / len(line))
    plt.title("{} match {} total outside error".format(np.round(1-error,2),
                                               np.round(total_error),2))

    plt.legend()
    plt.show()
    
variance = 1
test_9(variance)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
#
#
#
#                            Final reading code
#
#
#
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

#The upper,
U =
D =
B =

learned_bands = {'U': U, 'D': D, 'B': B}

def send_to_program(command):

    """The argument "command" will be a string and the output of this function
    will be either (1) No output, (2) Moving the slider up, (3) Moving the
    slider down, or (4) switching the button from on to off or vice versa

    Some test code you can play with and modify.  It generates a list of random
    numbers from 0 to 3 with certain probabilities and based on which number
    it gets it sends the appropriate command to the program.

    import numpy as np
    
    n_tests = 1000  #The number of tests you can run.
    zero_prob = 0.5  #The probability of getting a zero in the list
    other_prob = (1-zero_prob) / 3  #The other probabilities for the non-zeros
    test_commands = np.random.choice([0,1,2,3], n_tests,
    p = [zero_prob, other_prob, other_prob, other_prob])


    for i in range(n_tests):
        if test_commands[i] == 0:
            send_to_program('Do Nothing')
        if test_commands[i] == 1:
            send_to_program('U')
        if test_commands[i] == 2:
            send_to_program('D')
        if test_commands[i] == 3:
            send_to_program('B')

    """
    
    if command == 'Do Nothing':
        #Do nothing with the program, just wait for the next data command
        return 'Do Nothing'
    if command == 'U':
        #Move the slider's current position up
        return 'U'
    if command == 'D':
        #Move the slider's current position down
        return 'D'
    if command == 'B':
        #Switch the button from off to on or vice versa.
        return 'B'


def compare_signal_and_learned_bands(stream, learned_bands, args):
    """
    Takes a stream signal and compares it to each of the learned
    bands and returns the command with the highest match percentage.

    Parameters
    ----------

    stream : array_like
            Stream of signal data with number of channels specified in args.
    learned_bands : dictionary
            Dictionary of learned bands.
    args : dictionary
            Dictionary of all arguments.

    Returns
    -------

    command : string
            Command to be processed by the program that Stephen wrote
    """
    
    commands = ['U', 'D', 'B']
    #Get all the match percentages
    signal_match_percentages = [(1-compare_signal_and_band(stream,
                                        learned_bands[band_name])) \
                                for band_name in commands]

    if np.max(signal_match_percentages) < args['command_threshold']:
        command = "Do Nothing"
        return command

    command_index = np.argmax(signal_match_percentages)
    command = commands[command_index]
    return command


    

def compare_stream_data_to_bands(stream, learned_bands, args):
    """Final function that takes in the stream data and learned bands and
    sends the final band result to the program.

    The code is structured as follows:

    1. Get  the stream data.
    2. Compare it to the learned bands and choose a command.
    3. Send the command to the program.

    """

    #1. Get the stream data and put it in the right form.
    stream = pd.DataFrame(stream).T

    #2. Compare it to the set of bands and choose a command.
    
        #(Maybe run the time series through a collection of bands
        #and whichever one gets the highest probabiliy of being in
        #is returned)
        #How many bands do I need?  Three.  (1) Up slider (U), (2) Down slider
        #(D), and (3) Push button on/off (B).

        #Method 1: If Statemnets
    
            #Compare the live signal to each band with if statements
            #If no signal match % is above a certain threshold, then do nothing
            #Otherwise send the appropriate command over.
    
    command = compare_signal_and_learned_bands(stream, learned_bands)

    #3.  Send the command to the program
    #If the signal-scores for each band are less than the command threshold,
    #Then do nothing.
    send_to_program(command)
    
        #Method 2: Neural Network
        
        #Or use neural network to transform datastream to a band signal?
            #Signal can be transformed to (1) Do nothing, (2) Up/Down/Button
            #Need to record the data first to get a neural network
