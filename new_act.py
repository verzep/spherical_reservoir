from __future__ import absolute_import
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad as egrad

def new_activation(x):
    '''
    Rectifier
    '''

    o = np.ones(x.shape)
    v = x / np.maximum(o, x)
    return v

def new_activation2(x):
    '''
    Rectifier with non-linearity inside the unit sphere
    '''

    o = np.ones(x.shape)
    v = np.tanh(x / np.maximum(o, x))
    return v

def new_activation3(x):
    '''
    Smooth rectifier
    '''

    steepness = 10.0
    o = np.ones(x.shape)
    o = o * np.exp(1)
    v = steepness * x / np.log(o + np.exp(steepness * x))
    return v

# input range
x = np.linspace(0, 3, 1000)
        
plt.figure()
plt.subplot(311)
plt.title("Hard rectifier")
plt.plot(x, new_activation(x), label='Activation', linewidth=2.0)
plt.plot(x, egrad(new_activation)(x), label='Derivative', linewidth=2.0)
plt.legend(loc="best", prop={'size': 12})

plt.subplot(312)
plt.title("Hard non-linear rectifier")
plt.plot(x, new_activation2(x), label='Activation', linewidth=2.0)
plt.plot(x, egrad(new_activation2)(x), label='Derivative', linewidth=2.0)
plt.legend(loc="best", prop={'size': 12})

plt.subplot(313)
plt.title("Smooth rectifier")
plt.plot(x, new_activation3(x), label='Activation', linewidth=2.0)
plt.plot(x, egrad(new_activation3)(x), label='Derivative', linewidth=2.0)
plt.ylim((0, 1.1))
plt.legend(loc="best", prop={'size': 12})

plt.tight_layout()
plt.savefig("/home/lorenzo/newact_rectifier.pdf", format='pdf', dpi=1000, bbox_inches='tight', pad_inches=0.05)
plt.close()

