import numpy as np
from matplotlib import pyplot as plt


class ESN():
    def __init__(self, n_inputs=1, n_outputs=1, n_reservoir=1000,
                 spectral_radius=0.95, sparsity=0.9, noise=0.001,
                 input_shift=None, input_scaling=None,
                 output_feedback= False, output_scaling=None, output_shift=None,
                 regularization=0.001, online_training=False, learning_rate=0.01,
                 neuron_activation=0, leak_rate=1.0,
                 random_state=None, reservoir_uniform=True, transient=20, radius = 1, wigner = False):
        """
        Echo state network with leaky-integrator neurons.
        The network can operate both in generative and predictive mode.
        :param: n_inputs: nr of input dimensions
        :param: n_outputs: nr of output dimensions
        :param: n_reservoir: nr of reservoir neurons
        :param: spectral_radius: spectral radius of the recurrent weight matrix
        :param: sparsity: proportion of recurrent weights set to zero
        :param: noise: noise added to each neuron during training
        :param: input_shift: scalar or vector of length n_inputs to add to each
                    input dimension before feeding it to the network.
        :param: input_scaling: scalar or vector of length n_inputs to multiply
                    with each input dimension before feeding it to the netw.
        :param: output_feedback: if True, feed the target back into the dynamics
        :param: output_scaling: factor applied to the target/output signal
        :param: output_shift: additive term applied to the target/output signal
        :param: regularization: regularization coefficient for ridge regression
        :param: online_training: ridge regression or LMS training
        :param: learning_rate:
        :param: neuron_activation: 0=tanh with leaking neurons; 1=hard rectifier on the sphere;
                    2=hard rectifier with tanh; 3=soft rectifier; 4=like 1 with leaking neurons;
                    5 = like 1, but always projecting 7 = linear
        :param: leak_rate: parameter of leaky-integrator neurons
        :param: random_state: positive integer seed, np.rand.RandomState object,
                      or None to use numpy's built-in RandomState
        :param: reservoir_uniform: if True, use uniform random numbers, other use the standard normal distribution
        :param: transient: number of initial states to be discarded
        :param: radius: radius of the hyper-sphere
        :param: wigner: if true the reservoir connection are generated according to a GOE
        """

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise
        self.input_shift = self._correct_dimensions(input_shift, n_inputs)
        self.input_scaling = self._correct_dimensions(input_scaling, n_inputs)

        self.output_scaling = output_scaling
        self.output_shift = output_shift

        self.regularization = regularization
        self.online_training = online_training
        self.learning_rate = learning_rate
        self.leak_rate = leak_rate
        self.neuron_activation = neuron_activation
        assert neuron_activation in {0, 1, 2, 3, 4, 5,6,7}, 'Possible values: {0, 1, 2, 3, 4, 5,6,7} \'.'

        self.random_state = random_state
        self.reservoir_uniform = reservoir_uniform
        self.transient = transient
        self.radius = radius
        self.wigner = wigner

        # the given random_state might be either an actual RandomState object,
        # a seed or None (in which case we use numpy's builtin RandomState)
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.mtrand._rand

        self.output_feedback = output_feedback

        # init network parameters
        self._init_vars()

    def _init_vars(self):
        """
        Initialize all relevant variables
        """

        # initialize reservoir
        if self.reservoir_uniform is True:
            # weights uniformly distributed in [-1, 1]
            W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) * 2 - 1
            # delete a fraction of connections
            W[self.random_state_.rand(*W.shape) < self.sparsity] = 0.0

            if self.wigner == True:
                print 'WIGNER!'
                W = (W + W.T) / 2.0
            # scale the spectral radius of the reservoir
            radius = np.max(np.abs(np.linalg.eigvals(W)))
            self.W_res = W * self.spectral_radius / radius
        else:
            # random normally distributed matrix
            from math import sqrt
            W = self.random_state_.randn(self.n_reservoir, self.n_reservoir)
            # delete a fraction of connections
            W[self.random_state_.rand(*W.shape) < self.sparsity] = 0.0

            if self.wigner == True:
                print 'WIGNER!'
                W = (W+W.T)/2.0

            # scale the random matrix
            scale = float(self.spectral_radius) / sqrt((1.0 - self.sparsity) * self.n_reservoir)
            self.W_res = W * scale

        # random input weights uniformly distributed in [-1, 1]
        self.W_in = self.random_state_.rand(self.n_reservoir, self.n_inputs) * 2 - 1
        # random output feedback weights uniform in [-1, 1]
        self.W_fb = self.random_state_.rand(self.n_reservoir, self.n_outputs) * 2 - 1

        # read-out weights
        self.W_out = None
        # internal states
        self.states = None

    def _correct_dimensions(self, s, target_length):
        """
        Checks the dimensionality of some numeric argument s, broadcasts it
           to the specified length if possible.

        :param s: None, scalar or 1D array
        :param target_length: expected length of s
        :return: None if s is None, else numpy vector of length target_length
        """

        if s is not None:
            s = np.array(s)
            if s.ndim == 0:
                s = np.array([s] * target_length)
            elif s.ndim == 1:
                if not len(s) == target_length:
                    raise ValueError("arg must have length " + str(target_length))
            else:
                raise ValueError("Invalid argument")
        return s

    def _update(self, state, input, output):
        """
        Performs one update step, i.e., computes the next network state by applying the recurrent weights
        to the last state and feeding in the current input and, potentially, previous output.

        :param state: current network state
        :param input: next input
        :param output: current output
        :return: next network state
        """

        # state update
        pre_activation = np.dot(self.W_res, state) + np.dot(self.W_in, input)
        if self.output_feedback:
            pre_activation += np.dot(self.W_fb, output)

        # standard leaking neurons
        if self.neuron_activation == 0 or self.neuron_activation == 4:
            leak_term = (1.0 - self.leak_rate) * state
            noise_term = self.noise * (self.random_state_.randn(self.n_reservoir) * 2 - 1)
            # with tanh activation
            if self.neuron_activation == 0:
                new_state = leak_term + self.leak_rate * np.tanh(pre_activation) + noise_term
            # hard rectifier on linear pre-activation with leaking neurons
            else:
                norm_v = np.linalg.norm(pre_activation)
                if norm_v < 1:
                    norm_v = 1.0
                new_state = leak_term + self.leak_rate * (pre_activation / norm_v) + noise_term

        # hard rectifier on linear pre-activation
        elif self.neuron_activation == 1:
            noise_term = self.noise * (self.random_state_.randn(self.n_reservoir) * 2 - 1)

            norm_v = np.linalg.norm(pre_activation + noise_term)
            if norm_v < 1:
                norm_v = 1.0
            new_state = (pre_activation +noise_term) / norm_v
        # hard rectifier on tanh pre-activation
        elif self.neuron_activation == 2:
            # apply tanh directly to the current state
            pre_activation = np.dot(self.W_res, np.tanh(state)) + np.dot(self.W_in, input)
            if self.output_feedback:
                pre_activation += np.dot(self.W_fb, output)
            norm_v = np.linalg.norm(pre_activation)
            if norm_v < 1:
                norm_v = 1.0
            new_state = pre_activation / norm_v
        # soft rectifier on linear pre-activation
        elif self.neuron_activation == 3:
            norm_v = np.linalg.norm(pre_activation)
            steepness = 2.0
            r = np.log(np.exp(1) + np.exp(steepness * norm_v)) / steepness
            new_state = pre_activation / r

        # normalize in any case

        elif self.neuron_activation == 5:
            noise_term = self.noise * (self.random_state_.randn(self.n_reservoir) * 2 - 1)

            norm_v = np.linalg.norm(pre_activation + noise_term)
            #if norm_v < 1:
            #    norm_v = 1.0
            new_state = self.radius * (pre_activation + noise_term) / norm_v

        elif self.neuron_activation == 6:
            noise_term = self.noise * (self.random_state_.randn(self.n_reservoir) * 2 - 1)
            norm_v = np.linalg.norm(pre_activation + noise_term)
            # if norm_v < 1:
            #    norm_v = 1.0
            new_state = np.tanh(self.radius * (pre_activation + noise_term) / norm_v)

        elif self.neuron_activation == 7:
            noise_term = self.noise * (self.random_state_.randn(self.n_reservoir) * 2 - 1)

            new_state = pre_activation + noise_term

        return new_state

    def _scale_inputs(self, inputs):
        """
        For each input dimension j: multiplies by the j'th entry in the
        input_scaling argument, then adds the j'th entry of the input_shift
        argument.
        :param inputs: input signal as a vector
        """

        if self.input_scaling is not None:
            inputs = np.dot(inputs, np.diag(self.input_scaling))
        if self.input_shift is not None:
            inputs = inputs + self.input_shift

        return inputs

    def _scale_output(self, output):
        """
        Multiplies the target/output signal by the output_scaling argument,
        then adds the output_shift argument to it.
        :param: target/output signal used for training
        :return: scaled and shifted target/output signal
        """

        if self.output_scaling is not None:
            output = output * self.output_scaling
        if self.output_shift is not None:
            output = output + self.output_shift

        return output

    def _unscale_output(self, output_scaled):
        """
        Inverse operation of the _scale_output method.
        :param: scaled output signal
        :return: the original un-scaled output signal
        """

        if self.output_shift is not None:
            output_scaled = output_scaled - self.output_shift
        if self.output_scaling is not None:
            output_scaled = output_scaled / self.output_scaling

        return output_scaled

    def _ridge_regression(self, inputs, outputs):
        """
        Ridge regression optimization of read-out weights
        :param inputs: input signal
        :param outputs: output signal
        :return: optimized output weights with ridge regression and sequence of states on the training inputs
        """

        # generate the entire sequence of states
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        for n in range(1, inputs.shape[0]):
            states[n, :] = self._update(states[n - 1], inputs[n, :], outputs[n - 1, :])

        # ridge regression (discard some initial transient states)
        first = np.dot(states[self.transient:, :].T, outputs[self.transient:, :])
        second = np.linalg.pinv(np.dot(states[self.transient:, :].T, states[self.transient:, :])
                                + self.regularization * np.identity(states.shape[1]))
        w_out = np.dot(first.T, second)

        return w_out, states

    def _lms(self, inputs, outputs, mu=0.01):
        """
        least-mean-square online training of read-out weights
        :param inputs: input signal
        :param outputs: output signal
        :return: optimized output weights, sequence of states, and online predictions
        """

        # init read-out weights
        w_out = np.zeros((self.n_outputs, self.n_reservoir))
        # init sequence of states
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        # init predictions
        predictions = np.zeros((inputs.shape[0], self.n_outputs))
        predictions[0, 0] = outputs[0, 0]

        # time-steps
        for n in range(1, inputs.shape[0]):
            states[n, :] = self._update(states[n - 1], inputs[n, :], predictions[n - 1, :])
            # make prediction
            predictions[n, :] = np.dot(w_out, states[n, :])
            # error
            e = 0.
            for i in range(self.n_outputs):
                # error for output neuron i
                e = outputs[n, i] - predictions[n, i]
                # update weights
                w_out[i, :] = w_out[i, :] * (1 - mu * self.regularization) + (mu * states[n, :] * e)

        return w_out, states, predictions

    def get_internal_states(self):
        """
        Return the internal states of the network
        :return: the ESN states
        """

        return self.states

    def get_weight_matrices(self):
        """
        Return all weight matrices. Please note that W_out = [W_res_out, W_in_out]
        :return: all weight matrices
        """

        return self.W_res, self.W_in, self.W_fb, self.W_out

    def fit(self, inputs, outputs):
        """
        Learn the read-out weights.

        :param inputs: array of dimensions (N_training_samples x n_inputs)
        :param outputs: array of dimension (N_training_samples x n_outputs)
        :return The network's output on the training data, using the trained weights
        """

        # transform any vectors of shape (x, ) into vectors of shape (x, 1)
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))

        # transform input and output signals
        inputs_scaled = self._scale_inputs(inputs)
        outputs_scaled = self._scale_output(outputs)

        if self.online_training is False:
            # optimization via ridge regression
            self.W_out, self.states = self._ridge_regression(inputs_scaled, outputs_scaled)
            # produce predictions on training data
            pred_train = self._unscale_output(np.dot(self.states, self.W_out.T))
        else:
            self.W_out, self.states, pred_train = self._lms(inputs_scaled, outputs_scaled)

        # remember the last state, input, and output
        self.last_state = self.states[-1, :]
        self.last_input = inputs_scaled[-1, :]
        self.last_output = outputs_scaled[-1, :]



        return pred_train


    #This is only the evolution, with no training
    def evolve(self, inputs, outputs):

        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))

        # transform input and output signals
        inputs_scaled = self._scale_inputs(inputs)
        outputs_scaled = self._scale_output(outputs)

        # generate the entire sequence of states
        states = np.zeros((inputs_scaled.shape[0], self.n_reservoir))
        for n in range(1, inputs_scaled.shape[0]):
            states[n, :] = self._update(states[n - 1], inputs_scaled[n, :], outputs_scaled[n - 1, :])


        self.states = states

        self.last_state = self.states[-1, :]
        self.last_input = inputs_scaled[-1, :]
        self.last_output = outputs_scaled[-1, :]

        return states



    #this is only the training, once the evolution has been done
    def train(self, outputs, states = None):
        if states is None:
            states = self.states

        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))

        outputs_scaled = self._scale_output(outputs)

        first = np.dot(states[self.transient:, :].T, outputs_scaled[self.transient:, :])
        second = np.linalg.pinv(np.dot(states[self.transient:, :].T, states[self.transient:, :])
                                + self.regularization * np.identity(states.shape[1]))

        self.W_out = np.dot(first.T, second)

        return self._unscale_output(np.dot(self.states, self.W_out.T))






    def predict(self, inputs, continuation=True):
        """
        Apply learned model on test data.

        :param inputs: array of dimensions (N_test_samples x n_inputs)
        :param continuation: if True, start the network from the last training state
        :return predictions on test data
        """

        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))

        # set noise term to zero for state update during test
        self.noise = 0

        n_samples = inputs.shape[0]

        if continuation:
            last_state = self.last_state
            last_input = self.last_input
            last_output = self.last_output
        else:
            last_state = np.zeros(self.n_reservoir)
            last_input = np.zeros(self.n_inputs)
            last_output = np.zeros(self.n_outputs)

        inputs = np.vstack([last_input, self._scale_inputs(inputs)])
        states = np.vstack([last_state, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack([last_output, np.zeros((n_samples, self.n_outputs))])

        # process test set one sample at a time
        for n in range(n_samples):
            # next state
            states[n + 1, :] = self._update(states[n, :], inputs[n + 1, :], outputs[n, :])
            # predicted output
            outputs[n + 1, :] = np.dot(self.W_out, states[n + 1, :])

        # stack up new states
        self.states = np.vstack((self.states, states))



        return self._unscale_output(outputs[1:])



    def score(self, inputs, y=None):
        """
        To be used when extending scikit-learn base classifier
        :param inputs:
        :param y: targets
        :return: R2 score
        """
        from sklearn.metrics import r2_score
        return r2_score(self.predict(inputs), y)


    def plot_states(self, increments = False):
        plt.figure(figsize=( #self.states.shape[1] * 0.01,
            10
             ,self.states.shape[0] * 0.0025))
        if increments:


            incr = self.states[:] - np.mean(self.states[:], axis=0)
            plt.imshow(incr[200:], aspect='auto',
                       interpolation='nearest')
            plt.colorbar()
            plt.xlabel('neurons')
            plt.ylabel('time')



        else:
            plt.imshow(self.states[200:], aspect='auto',
                       interpolation='nearest')
            plt.colorbar()
            plt.xlabel('neurons')
            plt.ylabel('time')
