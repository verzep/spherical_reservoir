import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
from pyESN import ESN
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from copy import deepcopy

from data_mng import forecasting_data_split, memory_data_split

def simulate_error(x_train, y_train, x_test, y_test, scaling_param, SR_param
                   ,neuron_activation=0
                   ,num_sim=10):
    count = 0
    E_min = 100
    E_train_min = 100
    min_SR = None
    min_scal = None

    for SR in SR_param:
        #E_train = []
        #E_test = []
        #R2 = []
        #count += 1
        #print "SR ", SR

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
                r2 += r2_score(y_predicted[:], y_test[:])

            #print (mse_train / num_sim)

            #E_train.append(mse_train / num_sim)
            #E_test.append(mse / num_sim)
            #R2.append(r2 / num_sim)

            # selected with the training error
            if mse_train / num_sim < E_train_min:
                #print mse_train, ' < ', E_train_min
                #print 'I WAS ', E_min, "NOW ", mse / num_sim
                E_train_min = mse_train / num_sim
                E_min = mse / num_sim
                min_SR = SR / num_sim
                min_scal = scaling / num_sim

        #plt.plot(E_test)

    #plt.show()

    return E_min, min_SR, min_scal

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




#create random data
L = 20
tau_range = np.arange(0,L)
nu_range = np.logspace(-2.5,1,L)

scaling_param = np.linspace(0.01, 2, 20)
SR_param = np.linspace(0.2, 1.5, 20 )

all_error= []
all_SR = []
all_scal = []


for nu in nu_range:
    for tau in tau_range:
        print 'nu is ', nu, 'and tau is', tau
        x = np.random.rand(10000)*2 -1
        x_train, y_train, x_test, y_test = nonlinear_task(data=x,nu= nu, tau= tau)

        mse, best_SR, best_scal = simulate_error(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                                 neuron_activation=7,  # 0: tanh, 5: spherical, 7: linear
                                                 SR_param=SR_param,
                                                 scaling_param=scaling_param,
                                                 num_sim=1)

        print mse

        all_error.append(mse)
        all_SR.append(best_SR)
        all_scal.append(best_scal)

        np.save('E_test_7_part', np.array(all_error))
        np.save('SR_7_part', np.array(all_SR))
        np.save('scal_7_part', np.array(all_scal))

E_TEST = np.array(all_error).reshape(L, L)
SR = np.array(all_SR).reshape(L, L)
scal = np.array(all_scal).reshape(L, L)

np.save('E_test_7', E_TEST)
np.save('SR_7', SR)
np.save('scal_7 ', scal)

