#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy
import sklearn


# In[2]:


data=pd.read_csv('creditcard.csv')


# In[3]:


print(data.columns)


# In[4]:


print(data.shape)


# In[5]:


print(data.head)


# In[6]:



from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest


# In[7]:


data.describe()


# In[ ]:





# In[8]:


data=data.sample(frac=0.1, random_state=1)
print(data.shape)


# In[9]:


data.hist(figsize=(20,20))
plt.show()


# In[10]:


Fraud=data[data['Class']==1]
Valid=data[data['Class']==0]
outlier_fraction=len(Fraud)/float(len(Valid))
print(outlier_fraction)
print('Fraud Cases :{}'.format(len(Fraud)))
print('Valid Cases :{}'.format(len(Valid)))


# In[11]:


cormat=data.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(cormat,vmax=0.8,square=True)
plt.show()


# In[12]:


columns=data.columns.tolist()
columns=[c for c in columns if c not in ["Class"]]
target="Class"
X=data[columns]
Y=data[target]
print(X.shape)
print(Y.shape)


# In[13]:


from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
classifiers= {
    "Isolation Forest":IsolationForest(max_samples=len(X),contamination=outlier_fraction,random_state=1),
}


# In[14]:


n_outliers=len(Fraud)
for i , (clf_name,clf) in enumerate(classifiers.items()):
    if clf_name=="Isolation Forest":
        clf.fit(X)
        scores_pred=clf.decision_function(X)
        y_pred=clf.predict(X)
        
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    
    n_errors=(y_pred != Y).sum()
    
    print('{} : {} '.format(clf_name,n_errors))
    print('Accuracy: ',accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))
    


# In[ ]:





# In[ ]:





# In[26]:





# In[ ]:



    
    


# In[ ]:




