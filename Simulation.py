#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print('Python: {}'.format(sys.version))

import scipy
print('scipy: {}'.format(scipy.__version__))

import numpy
print('numpy: {}'.format(numpy.__version__))

import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))

import pandas
print('pandas: {}'.format(pandas.__version__))

import sklearn
print('sklearn: {}'.format(sklearn.__version__))

import skimage
print('skimage: {}'.format(skimage.__version__))


# In[2]:


from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from skimage.feature import hog
...


# In[4]:


...
url = "https://raw.githubusercontent.com/Santonu-dn/Simulation_Queueing_Systems-/main/dataset.csv"
names = ['customer no', 'interarivel time', 'arrivel time on clock']
dataset = read_csv(url, names=names)


# In[5]:


...
print(dataset.shape)


# In[6]:


...
print(dataset.head(6))


# In[7]:


...
dataset.hist()
pyplot.show()


# In[8]:


...
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()


# In[9]:


...
scatter_matrix(dataset)
pyplot.show()


# In[ ]:




