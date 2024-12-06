#!/usr/bin/env python
# coding: utf-8

# In[1]:


from heapq import nsmallest
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[2]:


df = pd.read_csv('weather_forecast_data.csv')
df


# # Task 1: Preprocessing

# ## Does the dataset contain any missing data? Identify them.

# In[3]:


df.info()


# In[4]:


df.isna().sum()


# In[5]:


df[df.isna().any(axis=1)]


# ## Splitting our data to training and testing for training and evaluating our models
# ## Apply the two techniques to handle missing data, dropping missing values and replacing them with the average of the feature.

# In[6]:


DROP_MISSING_VALUES = False

if DROP_MISSING_VALUES:
    dropped_missing = df.dropna()
    
    X = dropped_missing.drop(columns=['Rain'])
    Y = dropped_missing['Rain']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    
else:
    X = df.drop(columns=['Rain'])
    Y = df['Rain']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)


# ## Does our data have the same scale? If not, you should apply feature scaling on them.

# In[7]:


df.describe()


# In[8]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # Task 2: Implement Decision Tree, k-Nearest Neighbors (kNN) and naïve Bayes

# ## Implement k-Nearest Neighbors (kNN) algorithm from scratch.

# In[9]:


def knn_from_scratch(X_train, Y_train, point, k):
    def dist(neighbor):
        return ((point - neighbor[0]) ** 2).sum()
    
    nearest_k_neighbors = nsmallest(k, zip(X_train, Y_train), key=dist)
    counter = Counter(y for _, y in nearest_k_neighbors)
    return counter.most_common(1)[0][0]


# In[10]:


def eval_pred(pred):
    print('Accuracy:', accuracy_score(Y_test, pred))
    print('Precision:', precision_score(Y_test, pred, pos_label='no rain'))
    print('Recall:', recall_score(Y_test, pred, pos_label='no rain'))


# ## Scikit-learn Decision Tree

# In[11]:


print('Decision Tree')
decision_tree = DecisionTreeClassifier().fit(X_train, Y_train)
eval_pred(decision_tree.predict(X_test))


# ## Scikit-learn KNN

# In[12]:


print('KNN')
for k in range(3, 13, 2):
    print(f'k = {k}')
    eval_pred(KNeighborsClassifier(k).fit(X_train, Y_train).predict(X_test))


# ## KNN from scratch

# In[13]:


print('KNN from scratch')
for k in range(3, 13, 2):
    print(f'k = {k}')
    eval_pred([knn_from_scratch(X_train, Y_train, point, k) for point in X_test])


# ## Naïve Bayes

# In[14]:


print('Naïve Bayes')
eval_pred(GaussianNB().fit(X_train, Y_train).predict(X_test))


# # Task 3: Interpreting the Decision Tree and Evaluation Metrics Report

# In[15]:


plt.figure(figsize=(20, 20))

_ = plot_tree(
    decision_tree,
    feature_names=X.columns,
    class_names=['no rain', 'rain'],
    filled=True,
)

