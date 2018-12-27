def plot_fft(data):
    """
    :param data: data in time domain
    """
    from scipy.fftpack import fft
    import matplotlib.pyplot as plt
    import numpyt as np

    ts_len = data.shape[0]
    k = np.arange(ts_len)

    # sample spacing
    t = 1.0 / ts_len

    # sampling rate
    Fs = 150
    T = ts_len / Fs
    # two sides frequency range
    frq = k / T
    # one side frequency range
    frq = frq[range(ts_len // 2)]

    # FFT
    y = fft(data)
    #time = np.linspace(0.0, 1.0 / (2.0 * t), ts_len // 2)

    # Plot
    plt.figure(figsize=(15, 5))
    #plt.plot(frq, 2.0 / ts_len * np.power(np.abs(y[0:ts_len // 2]), 2), '-b')
    plt.plot(frq, np.abs(y[0:ts_len // 2]), '-b')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.show()


def autocorr(x):
    """
    Return the autocorrelation function
    :param x: input data
    :return: autocorrelation function
    """
    import numpy as np

    result = np.correlate(x, x, mode='full')
    return result[result.size//2:] / x.shape[0]


def get_jacobian(point, W_res, W_hat, alpha=1., feedback=True):
    """
    Linearization considering a leaking rate parameter
    :param point: point in phase space where to linearize
    :param W_res: recurrent connections matrix
    :param W_hat: perturbation matrix (e.g., from output feedback)
    :param alpha: leaking rate
    :param feedback: consider perturbation matrix or not
    :return: Jacobian matrix
    """

    import numpy as np

    if alpha > 1.:
        alpha = 1.
    elif alpha < 0.:
        alpha = 0.

    dim = point.shape[0]
    D = np.diag(np.ones(dim) - np.square(np.tanh(point)))
    J = (1. - alpha) * np.eye(dim)

    if feedback:
        J += alpha * np.dot(D, (W_res + W_hat))
    else:
        J += alpha * np.dot(D, W_res)

    return J


def unit_vector(vector):
    """
    Returns the unit vector of the input vector.
    """

    import numpy as np

    norm = np.linalg.norm(vector)
    tolerance = 1e-20

    if norm <= tolerance:
        return np.zeros((vector.shape[0]),)
    else:
        return vector / np.linalg.norm(vector)


def angle(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'.
    Norm of both v1 and v2 should be non-zero.
    """

    import numpy as np

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    if np.linalg.norm(v1_u) == 0. or np.linalg.norm(v2_u) == 0.:
        return 0.0
    else:
        return np.real(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
