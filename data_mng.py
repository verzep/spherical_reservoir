import numpy as np
from copy import copy, deepcopy
from matplotlib import pyplot as plt
import scipy.io as sio
from pyESN import ESN
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def spherical_coord(x):
    dim = len(x)
    t = np.zeros(dim)

    r = np.linalg.norm(x)
    for i in range(len(x)-2):
        D = np.linalg.norm(x[i:])
        t[i] = np.arccos(x[i]/D)

    t[-1]= np.sign(x[-1]) * np.arccos(x[-2]/np.linalg.norm(x[len(x)-2:] ))

    return  r,t


def simulate_error(x_train, y_train, x_test, y_test, scaling_param, SR_param
                   ,neuron_activation=0
                   ,num_sim=10):
    count = 0
    E_min = 100
    E_train_min = 100
    min_SR = None
    min_scal = None

    for SR in SR_param:
        for scaling in scaling_param:
            #print "scaling", scaling


            for sim in range(num_sim):
                mse_train = 0
                mse = 0
                r2 = 0

                esn = ESN(n_inputs=1,
                          n_outputs=1,
                          n_reservoir=200,
                          spectral_radius=SR,
                          radius=1,
                          sparsity=0.0,
                          noise=0.,
                          input_scaling=scaling / np.std(x_train),
                          output_scaling=scaling / np.std(y_train),
                          transient=200,
                          reservoir_uniform=True,
                          regularization=0.01,
                          online_training=False,
                          learning_rate=0,
                          neuron_activation=neuron_activation,
                          leak_rate=1,
                          # random_state=seed,
                          output_feedback=False
                          # , wigner = True
                          )

                transient = 50
                esn.evolve(x_train, y_train)
                y_train_predicted = esn.train(y_train)

                y_predicted = esn.predict(x_test)

                mse_train += np.sqrt(mean_squared_error(y_train_predicted[transient:], y_train[ transient:])) / np.std(y_train)
                mse += np.sqrt(mean_squared_error(y_predicted, y_test)) / np.std(y_test)
                #r2 += r2_score(y_predicted[:], y_test[:])

            #print (mse_train / num_sim)

            #E_train.append(mse_train / num_sim)
            #E_test.append(mse / num_sim)
            #R2.append(r2 / num_sim)

            # selected with the validation error
            if mse / num_sim < E_min:
                E_train_min = mse_train / num_sim
                E_min = mse / num_sim
                min_SR = SR / num_sim
                min_scal = scaling / num_sim

        #plt.plot(E_test)

    #plt.show()

    esn_f = ESN(n_inputs=1,
              n_outputs=1,
              n_reservoir=200,
              spectral_radius=min_SR,
              radius=1,
              sparsity=0.0,
              noise=0.,
              input_scaling=min_scal / np.std(x_train),
              output_scaling=min_scal / np.std(y_train),
              transient=200,
              reservoir_uniform=True,
              regularization=0.01,
              online_training=False,
              learning_rate=0,
              neuron_activation=neuron_activation,
              leak_rate=1,
              # random_state=seed,
              output_feedback=False
              # , wigner = True
              )

    transient = 50
    esn_f.evolve(x_train, y_train)
    y_train_predicted = esn_f.train(y_train)

    y_predicted = esn_f.predict(x_test)

    np.sqrt(mean_squared_error(y_train_predicted[transient:], y_train[transient:])) / np.std(y_train)
    mse_f = np.sqrt(mean_squared_error(y_predicted, y_test)) / np.std(y_test)
    #r2 += r2_score(y_predicted[:], y_test[:])


    return mse_f, min_SR, min_scal




def y_nonlinear(signal, nu, tau):
    arg = np.zeros(len(signal))

    for i in range(len(signal)):
        arg[i] = signal[i - tau]

    return np.sin(nu * arg[:])


def nonlinear_task(data, nu, tau, train_len=500, test_len=200):
    NL_data = y_nonlinear(data, nu, tau)

    x_train = data[2 * tau:train_len] + 0
    y_train = NL_data[2 * tau:train_len] + 0

    x_test = data[train_len:train_len + test_len] + 0
    y_test = NL_data[train_len:train_len + test_len] + 0

    return x_train, y_train, x_test, y_test










def forecasting_data_split(data, train_len, test_len, k_steps_ahead=1):
    """
    Split the input time series into train and test sets
    :param data: input data
    :param train_len: number of samples used for training
    :param test_len: number of samples used for testing
    :param k_steps_ahead: forecast horizon
    :return: x_train, y_train, x_test, y_test
    """

    x_train = copy(data[:train_len, ])
    y_train = copy(data[k_steps_ahead:(train_len + k_steps_ahead), ] )
    x_test = copy(data[train_len:(train_len + test_len - k_steps_ahead), ])
    y_test = copy(data[(train_len + k_steps_ahead):(train_len + test_len), ])

    return x_train, y_train, x_test, y_test


def memory_data_split(data, train_len, test_len, k_steps_behind=1):
    """
    Split the input time series into train and test sets
    :param data: input data
    :param train_len: number of samples used for training
    :param test_len: number of samples used for testing
    :param k_steps_behin: memory task length
    :return: x_train, y_train, x_test, y_test
    """

    x_train = copy(data[k_steps_behind:train_len + k_steps_behind, ])
    y_train = copy(data[:train_len, ] )

    x_test = copy(data[train_len + k_steps_behind:train_len + k_steps_behind + test_len, ] )
    y_test = copy(data[ train_len:train_len + test_len, ])

    return copy(x_train), copy(y_train), copy(x_test), copy(y_test)


def generator_data_split(data, train_len, generate_len):
    """
    Split data for generative mode
    :param data: input data
    :param train_len: number of samples used during training
    :param generate_len: length of data to be generated
    :return: x_train, y_train, x_test, y_test
    """

    import numpy as np

    # input for training is just a constant signal
    x_train = np.ones([train_len, 1])
    # target signal
    y_train = np.reshape(data[:train_len], [train_len, 1])
    # input driving the network during "free-running mode"
    x_test = np.ones([generate_len, 1])
    # ground truth for testing
    y_test = np.reshape(data[train_len:train_len+generate_len], [generate_len, 1])

    return x_train, y_train, x_test, y_test

def save_mg():
    import numpy as np

    data = np.load('mackey_glass_t17.npy')
    np.savetxt('../rnn-rp/mg.out', data, delimiter=',')


def gen_mso(length=2000, n_freq=8):
    """
    Generate a Multi-Superimposed Oscillator signal with six incommensurable frequencies
    and periods ranging from about 6 to about 120 discrete time steps
    :param length: Length of the signal
    :param n_freq: number of frequencies to consider (default 8)
    :return: The signal
    """

    from math import sin
    from sklearn import preprocessing

    signal = []
    freq = [0.2, 0.311, 0.42, 0.51, 0.63, 0.74, 0.85, 0.97]

    for n in range(length):
        s = 0.
        for f in freq[0:n_freq]:
            s += sin(n * f)
        signal.append(s)

    # rescale the signal
    signal = preprocessing.scale(signal)

    return signal


def write_mso(len=5000, n_freq=8):
    """
    Write MSO data on file
    :param len:
    """

    import numpy as np

    np.savetxt('./data/MSO_' + str(n_freq) + 'freq', gen_mso(len, n_freq), delimiter=',')


def write_square_wave(len=5000):
    """
    Write a square wave on file
    :param len: length of the signal (default 5000)
    """
    from scipy import signal
    import numpy as np

    t = np.linspace(0, 1, len, endpoint=False)
    s = signal.square(2 * np.pi * 100 * t)

    np.savetxt('./data/square_wave', s, delimiter=',')


def generate_nbit_flipflop(n=1, training_len=1000, test_len=50):
    """
    n-bit flip-flop problem for testing generative ESNs
    :param n: Number of bits to be considered
    :param training_len: Number of samples used for training
    :param test_len: Number of samples used for testing in generative mode
    :return: Input-Output pairs of training and test sets
    """

    import numpy as np

    # initialization
    total_len = training_len + test_len
    x = np.zeros([total_len, n])
    y = np.zeros([total_len, n])

    # first value
    e = np.random.randint(0, 2, size=[1, n])
    # -1 or 1
    r = np.zeros(e.shape)
    for i in range(n):
        r[0, i] = (-1) ** e[0, i]

    x[0, :] = r
    y[0, :] = r
    last_output_vec = r
    for i in range(1, total_len):
        # can be -1, 0, 1
        r_vec = np.random.randint(0, 3, size=[1, n]) - 1
        # input
        x[i, :] = r_vec

        # determine output independently for each bit
        for j in range(n):
            r = r_vec[0, j]
            last_output = last_output_vec[0, j]

            if r == 0 or r == last_output:
                current_output = last_output
            elif r != last_output:
                current_output = r
                last_output_vec[0, j] = current_output

            # output
            y[i, j] = current_output

    # training and test set split
    x_train = x[:training_len, :]
    y_train = y[:training_len, :]
    x_test = x[training_len:, :]
    y_test = y[training_len:, :]

    return x_train, y_train, x_test, y_test
