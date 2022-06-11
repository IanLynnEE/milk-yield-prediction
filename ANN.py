#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd

train = pd.read_csv('train_data.csv')


# In[2]:


train.loc[train['ranch']=='A', 'ranch'] = 1


# In[3]:


train.loc[train['ranch']=='B', 'ranch'] = 2


# In[4]:


train.loc[train['ranch']=='C', 'ranch'] = 3


# In[5]:


train = train.astype({"ranch": int})


# In[6]:


train=train.drop(['Unnamed: 0', 'ID'],axis=1)
train.info()


# In[7]:


correl = train.corr(method="spearman")


# In[8]:


correl


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(18,14))
heatmap = sns.heatmap(correl, vmin=-1, 
                      vmax=1, annot=True)


# In[10]:


#normalize
yorg_train = train['volume']
from sklearn import preprocessing
train= train.drop(['father', 'mother','birthday','lastBreed','SamplingDate','Testdate','inseminationDate',
                   'FinalInseminationSemen','FirstBreeding','FirstSemen'],axis=1)


# In[11]:


x = train.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
train=pd.DataFrame(x_scaled, columns=train.columns)
x_train = train.loc[:, train.columns != 'volume']
y_train = train['volume']


# In[12]:


from sklearn.model_selection import train_test_split
trainX, validX, trainY, validY = train_test_split(x_train, yorg_train, test_size = 0.2) #RMSE = 5.7129
# trainX, validX, trainY, validY = train_test_split(x_train, y_train, test_size = 0.2)  #RMSE = 0.0968


# In[13]:


from sklearn.neural_network import MLPRegressor
mlp_reg = MLPRegressor(hidden_layer_sizes=(150,100,50),
                       max_iter = 300,activation = 'relu',
                       solver = 'adam')

mlp_reg.fit(trainX,trainY)


# In[14]:


yv_pred = mlp_reg.predict(validX)


# In[15]:


from sklearn import metrics
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(validY, yv_pred)))


# In[16]:


plt.scatter(range(len(validY)), validY, c ="pink",
            linewidths = 2,
            marker ="s",
            edgecolor ="green",
            s = 50)
plt.scatter(range(len(yv_pred)), yv_pred, c ="yellow",
            linewidths = 2,
            marker ="^",
            edgecolor ="red",
            s = 200) 


# In[17]:


test = pd.read_csv('test_data.csv')


# In[18]:


test.loc[test['ranch']=='A', 'ranch'] = 1
test.loc[test['ranch']=='B', 'ranch'] = 2
test.loc[test['ranch']=='C', 'ranch'] = 3
test = test.astype({"ranch": int})
x_test= test.drop(['Unnamed: 0', 'ID','father', 'mother','birthday','lastBreed','SamplingDate','Testdate','inseminationDate',
                     'FinalInseminationSemen','FirstBreeding','FirstSemen','volume'],axis=1)
xx = x_test.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
xx_scaled = min_max_scaler.fit_transform(xx)
x_test=pd.DataFrame(xx_scaled, columns=x_test.columns)


# In[19]:


x_test.head()


# In[20]:


y_pred = mlp_reg.predict(x_test)


# In[21]:


print(y_pred)


# In[ ]:




