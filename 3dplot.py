
# coding: utf-8

# # Some data analysis stuff

# In[1]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D

states = np.loadtxt('../rnn-rp/states_MSO_gen.out', delimiter=',')

states_scaled = preprocessing.scale(states)

pca = PCA()
states_proj = pca.fit_transform(states_scaled)

# plot trajectory in 3D
start = 0
stop = 300
step = 5


fig = plt.figure(figsize=[15, 15])
ax = fig.gca(projection='3d')
ax.plot(states_proj[start:stop:step, 0], states_proj[start:stop:step, 1], states_proj[start:stop:step, 2], 'b^-')
ax.scatter(states_proj[start, 0], states_proj[start, 1], states_proj[start, 2], '^', s=30, c='r')
plt.title("3D Trajectory")
plt.show()


