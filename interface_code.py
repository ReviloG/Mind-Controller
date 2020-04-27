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

from scipy.fft import fft, ifft

from keras.layers import Input, Dense
from keras.models import Model

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
    samples = np.random.normal(args['mean'],args['std'],
                               (args['n_channels'], args['sample_size']))
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


def generate_noisy_sine(args, sine_args):
    A = sine_args[0]
    w = sine_args[1]
    theta = sine_args[2]
    
    samples = np.random.normal(args['mean'],args['std'],
                               (args['n_channels'], args['sample_size']))
    for i in range(args['n_channels']):
        sine = A * np.sin(w * np.linspace(0, 2 * np.pi, args['sample_size']) - \
                      theta)
        samples[i] = sine + samples[i]
    return samples

def generate_nn_samples(args, nn_samples, B_args, U_args, D_args):   
    
    button_samples = [generate_noisy_sine(args, B_args) for i in range(nn_samples)]
    slider_up_samples = [generate_noisy_sine(args, U_args) for i in range(nn_samples)]
    slider_down_samples = [generate_noisy_sine(args, D_args) for i in range(nn_samples)]

    data = {'B':button_samples, 'U': slider_up_samples,
              'D': slider_down_samples}

    
    return data

##def convert_data_to_nn_form(data):
##    """
##    data - dictionary, with U, M, B as keys representing buttons
##
##    returns raveled data in appropriate form
##    """
##    data_B = data['B']
##    data_U = data['U']
##    data_D = data['D']
##
##    data_B_ravel = [np.ravel(i) for i in data_B]
##    data_U_ravel = [np.ravel(i) for i in data_U]
##    data_D_ravel = [np.ravel(i) for i in data_D]
##
##    data_dict = {'B': data_B_ravel, 'U': data_U_ravel, 'D': data_D_ravel}
##
##    return data_dict

def stack_data(data_dict):
    """
    Get all the data from the dictionary and convert it to a stacked list.
    data_dict - dictionary, X_train data full of lists, keys 'B', 'U', 'D'
    """
    b = np.vstack([v for v in data_dict['B']])
    u = np.vstack([v for v in data_dict['U']])
    d = np.vstack([v for v in data_dict['D']])
    stacked_array = np.vstack((b,u,d))
    return stacked_array, len(b), len(u), len(d)
    

def get_data_and_labels(data):
    """
    1) Put the data into a stacked form
    2) Get the labels
    3) Randomize the data and labels
    4) Return the randomized data and labels.

    Arguments:
    
        data - (dictionary), X_train data full of lists, keys 'B', 'U', 'D'
    
    Returns:

        stacked_array - (list), All the data put into the appropriate form
        labels - (list), All the labels put into the appropriate form.

    """

    #1) Put the data inot a stacked form
    stacked_array, b_len, u_len, d_len  = stack_data(data)

    #2) Get the labels

    #B = 0, U = 1, D = 2
    lb = np.zeros(b_len).astype(int)
    lu = 1 * np.ones(u_len).astype(int)
    ld = 2 * np.ones(d_len).astype(int)

    labels = np.concatenate((lb, lu, ld))

    #3) Randomize the data and labels
    indices = np.arange(stacked_array.shape[0])
    np.random.shuffle(indices)

    stacked_array = stacked_array[indices]
    labels = labels[indices]

    #4) Return the randomized data and labels.
    return stacked_array, labels

def make_network(args):

    input_tensor = Input(shape=(args['sample_size'] * args['n_channels'],))
    x = Dense(32, activation='relu')(input_tensor)
    x = Dense(32, activation='relu')(x)
    output_tensor = Dense(2, activation='softmax')(x)
    model = Model(input_tensor, output_tensor)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def predict(data_points, model):
    """

    Function that gets the percentage match of a list of data points w/ a model.
    
    Arguments:
    ----------

        data_points - (numpy_array), list of shape (n, n_channels X sample_size)
        model - (keras model), model to make the prediction with.

    Returns:
    --------

        preiction - (numpy_array), array that contains the prediction match.
    
    """
    if data_points.shape == (8000,):
        data_points = data_points.reshape((1,8000))
        
    prediction = model.predict(data_points)    
    return prediction

def get_predictions(pred_data, models, args):
    """

    The function takes in data to be predicted
    and returns an array of the final predictions.  The function is set
    up such that it can take in a single data-stream measurement (8-channels
    stacked horizontally) or multiple data-stream measurements.

    If you wanted to run this for live data, you would use a
    single data-stream of the shape mentioned above.

    If you wanted to run this for non-live data, you would use multiple
    data-streams and pred_data would be of shape:
    [number_of_data_streams, (n_channels * sample_size)].

    
    Arguments:
    ----------

        pred_data - (numpy array), shape N X (n_channels* sample_size)
        models - (dictionary), dictionary of keras neural networks
        args - (dictionary), dictionary of all the arguments

    Returns:
    --------

        pred_args - (numpy array), array of shape (len(pred_data),).  It
                                    is an array of the final predictions.
    """

    preds_B = predict(pred_data, models['B'])[:,1]
    preds_U = predict(pred_data, models['U'])[:,1]
    preds_D = predict(pred_data, models['D'])[:,1]
    preds = np.vstack([preds_B, preds_U, preds_D])

    pred_args = []
    cols_over_thresh = np.max(preds, axis = 0) > args['nn_thresh']
    for i,v in enumerate(cols_over_threh):
        if v == True:
            pred_args.append(np.argmax(preds[:, i]))
        if v == False:
            pred_args.append(3)
            
    return pred_args

def convert_preds_to_commands(predictions):
    commands = []
    for i in range(len(predictions)):
        if predictions[i] == 0:
            commands.append('B')
        if predictions[i] == 1:
            commands.append('U')
        if predictions[i] == 2:
            commands.append('D')
        if predictions[i] == 3:
            commands.append('Do nothing')
    return commands


def generate_sample_train_test():
    X_train_gen = {x:np.random.randint(16,
                size = (300, 8000)) for x in ['B', 'U', 'D']}
    
    y_train_gen = {}
    names = ['B', 'U', 'D']
    for i in range(3):
        y_train_gen[names[i]] = np.random.choice([0,1],300)

    return X_train_gen, y_train_gen

def separate_labels_into_BUD(X_train, y_train):
    X_B = X_train['B']
    X_U = X_train['U']
    X_D = X_train['D']

    y_B = 1 * y_train['B']
    y_U = 2 * y_train['U']
    y_D = 3 * y_train['D']

    X = np.vstack([X_B, X_U, X_D])
    y = np.concatenate([y_B, y_U, y_D])

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    y_train_B = 1 * (y == 1)
    y_train_U = 1 * (y == 2)
    y_train_D= 1 * (y == 3)

    return X, y_B_train, y_U_train, y_D_train

def train_networks(X, y_train_B, y_train_U, y_train_D, networks, epochs):
    networks['B'].fit(X, y_train_B, epochs = epochs)
    networks['U'].fit(X, y_train_U, epochs = epochs)
    networks['D'].fit(X, y_train_U, epochs = epochs)


def train_networks_final(X_train, y_train, networks, n_epochs):
    X, y_train_B, y_train_U, y_train_D  = separate_labels_into_BUD(X_train,
                                                                   y_train)
    train_networks(X, y_train_B, y_train_U, y_train_D, networks, n_epochs)
    
def test_networks_final(pred_data, networks, args):
    #0 is Button, 1 is Up, 2 is Down
    final_predictions = get_predictions(pred_data, networks, args)

    #Get the commands associated with the final_predictions
    commands = convert_preds_to_commands(final_predictions)
    return commands

def convert_data_to_nn_form(data, args):
    """

    Function that converts the data to the
    appropriate form for the neural networks

    Arguments:
    ----------

        data : (numpy array), an array of the form
                                (n_data_points, n_channels, sample_size)
                                or of the form (n_channels, sample_size).
        args : (dictionary), the dictionary of the arguments

    Returns:
    --------

        data : (numpy array), the above data array reshaped to be of the form
                            (n_data_points, n_channels * sample_size)
    
    """

    if len(data.shape) == 3:
        data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        return data
    if len(data.shape) == 2:
        data = data.reshape(1, data.shape[0] * data.shape[1])
        return data


################################################################################
################################################################################
#
#                              Set Up
#
################################################################################
################################################################################

# Set up the data

#Import all the functions from the location the file:
# "interface_code_functions.py" is stored

import os
directory = 'c://Users/Oliver/Desktop/H'+\
            'omework/2020/Spring 2020/Senior_design_2'
os.chdir(directory)


#8 channels, 1000 images, 0 mean for the noise, standard deviation 1,
#command_threshold is the % that must be crossed to be considered correct.

args = name(n_channels = 8, sample_size = 1000, mean = 0, std = 1,
            command_threshold = 0.50, nn_thresh = 0.50)



#Set window and number of standard deviations for the Bollinger Bands
w_size = 8
n_stds = 2


#Create test band data
bands = generate_n_bollinger_bands(args, 3, w_size, n_stds)

#Create test stream data
stream = to_dataframe(args, generate_samples_2(args))
command = compare_stream_to_bands(stream, bands, args)

#send_to_program(command)

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



################################################################################
################################################################################
#
#                              Neural Network Test
#
################################################################################
################################################################################

###These are the arguments for the amplitude, frequency, and offset
###For generating the dataset
##B_args = [10, 3, 45]
##U_args = [8, 5, 22.5]
##D_args = [6, 2, 90]
##
###Dictionary containing all generated samples for the nn test
###Create a dictionary that contains lists of 100 samples of each B,U, and D.
###X_train['B'] is a list
##nn_samples = 100
##X_train = generate_nn_samples(args, nn_samples, B_args, U_args, D_args)
##
##X_train_good = convert_data_to_nn_form(X_train, args)
##X_train_final, labels = get_data_and_labels(X_train_good)
##

#Make the models
model_B = make_network(args)
model_U = make_network(args)
model_D = make_network(args)

networks = name(B = model_B, U = model_U, D = model_D)

#Create the network dictionary and epoch numbers.
networks = name(B = model_B, U = model_U, D = model_D)
n_epochs = 10

X_train, y_train = generate_sample_train_test()

#Part 2: Train the networks.

#X_train must be a dictionary (keys: "B", "U", "D") of numpy arrays of shape
#(m, n_channels * sample_size), where "m" is the number of data points
#that were collected. y_train is a dictionary (keys: "B", "U", "D") of
#numpy arrays of shape (m,) where "m" is the number of data points.
train_networks_final(X_train, y_train, networks, n_epochs)

#Part 3: Test data on the networks.

#Prediction data must have shape (m, n_channels * sample_size) or
#of shape (n_channels, sample_size)
prediction_data = np.random.randint(16, size = (3, 8, 1000))
pred_data = convert_data_to_nn_form(prediction_data, args)
commands = test_networks_final(pred_data, networks, args)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
#
#
#
#                            Tests
#
#
#
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

###(1.) Get sample sine waves and plot them
##samples = generate_samples(args)
##plot_samples(args, samples)
##
##
###(2.) Setting up the bands and plot the moving average
##test = np.random.randn(args['sample_size'])
##moving_average_list = [2*n+2 for n in range(10)]
##plot_bollinger(args, samples[-1], moving_average_list)
##plot_bollinger_all(args, samples[-1], moving_average_list)
##
##
##
###(3.) Get test data and plot the bollinger bands for the data
##test = np.random.randn(args['sample_size'])
##data = pd.DataFrame(test)
##
##samples = generate_samples(args)
##data2 = pd.DataFrame(samples).T
##
##x, z, y = Bollinger_Bands(data2, w_size = 8, n_stds = 2)
##
##plot_Bollinger_function(data, w_size = 8, n_stds = 2)
##
##
##
###(4.) Generate sample sine waves and plot their Bollinger bands with
##      #window size 8 and std 2
##samples = generate_samples(args)
##data2 = to_dataframe(args, samples)
##n_stds = 2
##w_size = 8
##plot_bollinger_bands(data2, args, w_size = w_size, n_stds = n_stds)
##
###(5.) Create list of DataFrames that represent signal data
##      #Plot the 0th DataFrame
##data = [to_dataframe(args, generate_samples(args)) for i in range(100)]
##plot_bollinger_bands(data[0], args, w_size = 8, n_stds = 2)
##
##
##
###(6.)  #Create a signal, calculate all its bands, and plot it
##       #with the bands.  Plot it with a disturbance.
##signal = pd.DataFrame(np.random.randn(1,1000) + \
##                           5 * np.sin(np.linspace(0, 10 * np.pi, 1000))).T
##rm, ub, lb = plot_single_Bollinger(signal, w_size = 4, n_stds = 2)
##
###Create a dictionary of bands
##bands= {'ub':ub, 'lb':lb, 'rm':rm}
##
###Calculate the error for the test code created three lines ago.
###Added +1 to the signal as a test to see how the error function would work.
##disturbance = 1
##
##error = compare_signal_and_band(signal + disturbance, bands)
##
##plot_signal_with_bands(signal, bands, disturbance)
##
###(7.) Plot the errors to see where they occur the most.
##noise = np.random.randn(1, 1000)
##noise = pd.DataFrame(noise).T
##upper_error, lower_error = plot_errors(signal + noise, bands)
##
###(8.) Simulate a stream and 3 bands and compare them to each band, plot the
##      #comparisons with error percentages.
##
###Set window and number of standard deviations for the Bollinger Bands
##w_size = 8
##n_stds = 2
##
##
###Create band data
##bands = generate_n_bollinger_bands(args, 3, w_size, n_stds)
##
###Create the stream data
##stream = to_dataframe(args, generate_samples_2(args))
##command = compare_stream_to_bands(stream, bands, args)
##
##
##
####x = []
####for i in range(100):
####    stream = to_dataframe(args, generate_samples_2(args))
####    command = compare_stream_to_bands(stream, bands, args)
####    x.append(command)
####    print(i/100)
####x = np.array(x)
####x[x=='U'] = 0
####x[x=='D'] = 1
####x[x=='B'] = 2
####x[x=='Do nothing'] = 4
####x = x.astype(np.int)
####np.histogram(x)
##
##
##plot_bands_and_stream_together(stream, bands, args)
##            
##plot_bands_and_stream(stream, bands, args)
##
##
##
### (9.) Find a better similarity check between two signals.  Create simple
###       sine wave bounds and add noise to a sine signal and plot the
###       results and get the error
##
##def test_9(variance):
##    base = np.sin(3 * np.linspace(0, 2* np.pi, 1000))
##    a = 1 + base
##    b = a - 2
##    c =  base + np.random.normal(0,variance,1000)
##    line = np.zeros((len(a), 1))
##
##    plt.plot(a, label='upper', color='blue')
##    plt.plot(b, label='lower', color = 'red')
##    plt.plot(c, label = 'data', color='green')
##    plt.plot(line, color='black')
##
##    above = (c > a)
##    below = (c < b)
##    upper_error = c[above] - a[above]
##    lower_error = b[below] - c[below]
##    total_error = np.sum(upper_error) + np.sum(lower_error)
##
##
##    outside = (c > a) | (c < b)
##    error = np.sum(outside / len(line))
##    plt.title("{} match {} total outside error".format(np.round(1-error,2),
##                                               np.round(total_error),2))
##
##    plt.legend()
##    plt.show()
##    
##variance = 1
##test_9(variance)


################################################################################
################################################################################
#
#                              FFT Test
#
################################################################################
################################################################################


from scipy.fft import fft, ifft

#Take the fft of each column in the dataframe, "axis=0" specifies this.
#Do the same for each band
stream_fft = fft(stream, axis = 0)
bands_fft = {key:{i:fft(bands[key][i], axis = 0) for i in range(len(bands))} \
             for key in bands.keys()}

fig, ax = plt.subplots(8,1)
for i in range(8):
    ax[i].plot(stream_fft[:, i])
plt.show()

a = np.sin( np.linspace(0,2*np.pi, 1000)) + np.cos( np.linspace(0,2*np.pi, 1000))
##noise = np.random.normal(0,1, 1000)
##a = a + noise
b = fft(a)

fig, ax = plt.subplots(2,1)
ax[0].plot(a)
ax[1].plot(np.real(b))

plt.show()

c = generate_samples_2(args)
c = to_dataframe(args, c)
fig, ax = plt.subplots(8,1)
for i in range(8):
    ax[i].plot(fft(c, axis = 0)[:, i])
    ax[i].plot(c.iloc[:, i])

plt.show()
