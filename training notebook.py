
# coding: utf-8

# In[1]:


import network
import helpers
import numpy as np


# In[2]:


import matplotlib.pyplot as plt


# In[9]:


img, lbl = helpers.load_training_data()
X, Y = helpers.convert_to_xy(img, lbl)
X_train, Y_train, X_test, Y_test = helpers.create_train_set(X,Y)
learning_rate = 0.01
layer_dims = [X.shape[0], 100, 20, 30, Y.shape[0]]
net = network.Network(layer_dims, learning_rate)
costs = net.train(X_train, Y_train, 1000)


# In[10]:


plt.plot(costs)
plt.show()


# In[13]:


costs = net.train(X_train, Y_train, 1000)


# In[14]:


network.compute_accuracy(net, X_train, Y_train)


# In[15]:


network.compute_accuracy(net, X_test, Y_test)


# In[ ]:


img, lbl = helpers.load_test_data()
X_val, Y_val = helpers.convert_to_xy(img, lbl)
network.compute_accuracy(net, X_val, Y_val)

