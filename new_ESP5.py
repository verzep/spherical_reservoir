import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
from pyESN import ESN
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from copy import deepcopy

from data_mng import *

#create random data
L = 20
tau_range = np.arange(0, L)
nu_range = np.logspace(-2.5,1,L)

scaling_param = np.linspace(0.01, 2, 20)
SR_param = np.linspace(0.2, 10, 20 )
all_error= []
all_SR = []
all_scal = []


for nu in nu_range:
    for tau in tau_range:
        print 'nu is ', nu, 'and tau is', tau
        x = np.random.rand(10000)*2 -1
        x_train, y_train, x_test, y_test = nonlinear_task(data=x,nu= nu, tau= tau)

        mse, best_SR, best_scal = simulate_error(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                                 neuron_activation=5,  # 0: tanh, 5: spherical, 7: linear
                                                 SR_param=SR_param,
                                                 scaling_param=scaling_param,
                                                 num_sim=1)

        print mse

        all_error.append(mse)
        all_SR.append(best_SR)
        all_scal.append(best_scal)

        np.save('E_test_5_part', np.array(all_error))
        np.save('SR_5_part', np.array(all_SR))
        np.save('scal_5_part', np.array(all_scal))


E_TEST = np.array(all_error).reshape(L, L)
SR = np.array(all_SR).reshape(L, L)
scal = np.array(all_scal).reshape(L, L)

np.save('E_test_5', E_TEST)
np.save('SR_5', SR)
np.save('scal_5', scal)

