#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotly
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from pandas.plotting import autocorrelation_plot, lag_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plotly.__version__
import missingno as msno
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import r2_score
import scipy.stats
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import HTML
from datetime import date


# In[2]:


def start_plot(figsize=(10, 8), style = 'whitegrid', dpi=100):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(1,1)
    plt.tight_layout()
    with sns.axes_style(style):
        ax = fig.add_subplot(gs[0,0])
    return ax


# In[3]:


"""Read the data"""

data = pd.read_csv('./data/report.csv',index_col=None, parse_dates=[8,12,13,15,16,19,20])
spec = pd.read_csv('./data/spec.csv', index_col=None)
birth = pd.read_csv('./data/birth.csv', index_col=None)
breed = pd.read_csv('./data/breed.csv', index_col=None)
cols = [ 'ID', 'year', 'month', 'ranch', 
            'serial', 'father','mother', 'birthday', 
            'delivery', 'lactation', 'volume', 'lastBreed',
            'SamplingDate', 'age', 'Testdate', 'inseminationDate', 
           'FinalInseminationSemen', 'BreedingTimes', 'LastDelivery','FirstBreeding','FirstSemen']
data.columns = cols

#transfrom time
data['birthday'] = pd.to_datetime(data['birthday'], format='%Y/%m/%d 00:00')
data[ 'lastBreed'] = pd.to_datetime(data[ 'lastBreed'], format='%Y/%m/%d 00:00')
data['Testdate'] = pd.to_datetime(data['Testdate'], format='%Y/%m/%d 00:00')
data['LastDelivery'] = pd.to_datetime(data['LastDelivery'], format='%Y/%m/%d 00:00')

msno.bar(data)


# In[4]:


data.info()


# In[5]:



"""Fill the missing values"""
#Set the complement value of NaN in Father cow semen number to '外購'
fbsn = data[data['father'].isnull()]
data['father'] = data['father'].fillna('Outsourcing')

#Set the complement value of NaN in Mother cow number to '外購'
mcn = data[data['mother'].isnull()]
data['mother'] = data['mother'].fillna('Outsourcing')

#Set the complement value of the Lactation days to its average
mean_ld = data['lactation'].mean()
data['lactation'] = data['lactation'].fillna(mean_ld)

#We found (Sampling date - Last date of birth) == Lactation days
#Set the complement value of the Last date of birth to (Sampling date - Lactation days)
data['lastBreed'] = data['SamplingDate']-data['lactation']*np.timedelta64(1,'D')


# In[6]:


data['lastBreed']


# In[7]:


"""Analysis"""
#Interval of Last breeding date and Date of first breeding v.s Lactation days
breeding_int =  (data['inseminationDate'] - data['FirstBreeding']) / np.timedelta64(1,'D')
data['breeding interval'] = breeding_int
li = data.groupby(['breeding interval']).mean()[ 'lactation'].reset_index()
ax = start_plot(style='darkgrid')
sns.regplot(li[ 'lactation'], li['breeding interval'], ax=ax)


# In[8]:


#Relationship between Milk volume and last breeding date
lbd = data.groupby(['inseminationDate']).mean()['volume'].reset_index()
lbd = lbd.set_index('inseminationDate')
lbd.dropna(axis=0, inplace=True)

#Relationship between Milk volume and Date of first breeding
dofb = data.groupby(['FirstBreeding']).mean()['volume'].reset_index()
dofb = dofb.set_index('FirstBreeding')
dofb.dropna(axis=0, inplace=True)

#Relationship between Milk volume and Last date of birth
ldob = data.groupby(['lastBreed']).mean()['volume'].reset_index()
ldob = ldob.set_index('lastBreed')
ldob.dropna(axis=0, inplace=True)

#Relationship between Milk volume and Sampling date
sd = data.groupby(['SamplingDate']).mean()['volume'].reset_index()
sd = sd.set_index('SamplingDate')
sd.dropna(axis=0, inplace=True)

fig, ax = plt.subplots(2,2, figsize=(10, 10), dpi=200)

window = 30
sd_window = 3
ax[0][0].plot(lbd['volume'].rolling(window=window).mean(), label='Last breeding date MA=%d' % window, color='navy')
ax[0][0].legend(shadow=True)
ax[0][1].plot(dofb['volume'].rolling(window=window).mean(), label='Date of first breeding MA=%d' % window, color='darkorange')
ax[0][1].legend(shadow=True)
ax[1][0].plot(ldob['volume'].rolling(window=window).mean(), label='Last date of birth MA=%d' % window, color='teal')
ax[1][0].legend(shadow=True)
ax[1][1].plot(sd['volume'].rolling(window=sd_window).mean(), label='Sampling date MA=%d' % sd_window, ls='--', color='red')
ax[1][1].legend(shadow=True)

lag1 = sd['volume'].diff(1)[1:].dropna(axis=0)
fig, ax = plt.subplots(2, 1, figsize=(12, 7), dpi=120, sharex=True)
plot_acf(lag1, lags=30, ax=ax[0])
plot_pacf(lag1, lags=30, ax=ax[1])
plt.show()


# In[9]:


"""Fill Date of first breeding & Last breeding date"""

#Interval of Sampling date and Last breeding date v.s Lactation days
d_int = (data['SamplingDate']-data['inseminationDate'])/np.timedelta64(1,'D')

#Interval of Sampling date and Date of first breeding v.s Lactation days
e_int = (data['SamplingDate']-data['FirstBreeding'])/np.timedelta64(1,'D')

#Interval of Last breeding date and Last date of birth v.s Lactation days
g_int = (data['inseminationDate']-data['lastBreed'])/np.timedelta64(1,'D')

#Interval of Date of first breeding and Last date of birth v.s Lactation days
h_int = (data['FirstBreeding']-data['lastBreed'])/np.timedelta64(1,'D')

data['d_int'] = d_int
data['e_int'] = e_int
data['g_int'] = g_int
data['h_int'] = h_int

di = data.groupby(['lactation']).mean()['d_int'].reset_index()
ei = data.groupby(['lactation']).mean()['e_int'].reset_index()
gi = data.groupby(['lactation']).mean()['g_int'].reset_index()
hi = data.groupby(['lactation']).mean()['h_int'].reset_index()

ax = start_plot(figsize=(8, 6), style='darkgrid', dpi=200)
sns.regplot(di['lactation'].values, di['d_int'].values, label='Sampling date - Last breeding date', color='teal', scatter_kws={'s':2})
sns.regplot(ei['lactation'].values, ei['e_int'].values, label='Sampling date - Date of first breeding', color='darkorange', scatter_kws={'s':2})
sns.regplot(gi['lactation'].values, gi['g_int'].values, label='Last breeding date - Last date of birth', color='brown', scatter_kws={'s':2})
sns.regplot(hi['lactation'].values, hi['h_int'].values, label='Date of first breeding - Last date of birth', color='navy', scatter_kws={'s':2})

ax.set_ylabel('Time Interval')
ax.set_xlabel('Lactation days')
ax.axvline(x=270, ls='--', c='y')
ax.legend(shadow=True)


# In[10]:


#Interval of Sampling date and Date of first breeding v.s Lactation days can be fitted into Linear regression
ei.dropna(axis=0, inplace=True)
Xadd = sm.add_constant(ei['lactation'])
model = sm.OLS(ei['e_int'], Xadd).fit()
b, a = model.params

#Fill in Date of first breeding
t_copy = data[data['FirstBreeding'].isnull()].copy()
t_copy['e_int'] = a * t_copy['lactation'] + b
t_copy['FirstBreeding'] = t_copy['SamplingDate']-t_copy['e_int']*np.timedelta64(1,'D')
t_copy['FirstBreeding'] = t_copy['FirstBreeding'].dt.date
t_copy['FirstBreeding'] = pd.to_datetime(t_copy['FirstBreeding'])
data.loc[t_copy.index, 'FirstBreeding'] = t_copy['FirstBreeding']


# In[11]:


#It is found that the feature missing value index of these four features has many the same
qlbd_idx = data[data['inseminationDate'].isnull()].index
qlis_idx = data[data['FinalInseminationSemen'].isnull()].index
qdofb_idx = data[data['FirstBreeding'].isnull()].index
qfis_idx = data[data['FirstSemen'].isnull()].index
s = list(set(qlbd_idx) & set(qlis_idx) & set(qdofb_idx) & set(qfis_idx))
print('These features has inner null data: %s' % len(s))
data.drop(s, axis=0, inplace=True)

#But the Last breeding date, Last insemination semen still have null
qlbd_idx2 = data[data['inseminationDate'].isnull()].index
qlis_idx2 = data[data['FinalInseminationSemen'].isnull()].index
s2 = list(set(qlbd_idx2) & set(qlis_idx2))
print('These features has inner null data: %s' % len(s2))
data.drop(s2, axis=0, inplace=True)

#Drop Date of last birth: too many nulls (>50%)
data.drop('LastDelivery', axis=1, inplace=True)


# In[12]:


#train_data&test_data
n_idx = data[data['volume'].isna()].index
train_data = data.drop(index=n_idx)
test_data = data[data['volume'].isna()]
fig, ax = plt.subplots(2, 1, figsize=(9, 6), dpi=200, sharex=True)
msno.bar(train_data, ax = ax[0], fontsize=7)
msno.bar(test_data, ax = ax[1], fontsize=7)


# In[13]:


#Only "Lactation days" is Normal distribution
ax = start_plot(figsize=(8, 6), style='darkgrid')
sns.kdeplot(train_data['lactation'], shade=True, label='Train %s' % len(train_data))
sns.kdeplot(test_data['lactation'], shade=True, label='Test %s' % len(test_data))
ax.legend(shadow=True)


# In[14]:


"""Correlation"""
train_data['delivery'] = train_data['delivery'].astype(int)
train_data['age'] = train_data['age'].astype(int)
num_features = train_data.select_dtypes(include=np.number).columns.tolist()
num_features.remove('ID')
num_features.remove('serial')
num_features.remove('breeding interval')
num_features.remove('d_int')
num_features.remove('e_int')
num_features.remove('g_int')
num_features.remove('h_int')

fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
corr = train_data[num_features].corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool)) # triangular (half of the matrix)
sns.heatmap(train_data[num_features].corr(), annot=True, cmap='RdBu_r', mask=mask, fmt='.2f', linewidth=0.8, ax=ax)


# In[15]:


train_data.to_csv('train_data.csv')
test_data.to_csv('test_data.csv')


# In[16]:


train_data[num_features].corr()


# In[ ]:




