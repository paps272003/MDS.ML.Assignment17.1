
# coding: utf-8

# In[16]:

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

url="https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv"

titanic = pd.read_csv(url)
titanic.head()


# In[4]:

titanic.columns =['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']


# In[5]:

#We will be using Pclass, Sex, Age, SibSp (Siblings aboard), Parch (Parents/children aboard), and Fare to predict whether a passenger survived.
titanic = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]


# In[6]:

#We need to convert ‘Sex’ into an integer value of 0 or 1.
titanic['Sex'] = titanic['Sex'].map({'male': 0, 'female': 1})


# In[7]:

#We will also drop any rows with missing values.
titanic = titanic.dropna()


# In[8]:

X = titanic.drop('Survived', axis=1)
y = titanic['Survived']


# In[12]:

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model = tree.DecisionTreeClassifier()
model


# In[13]:

#we fit our model using our training data
model.fit(X_train, y_train)


# In[15]:

#Then we score the predicted output from model on our test data against our ground truth test data.
y_predict = model.predict(X_test)
accuracy_score(y_test, y_predict)


# In[17]:

pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted Not Survival', 'Predicted Survival'],
    index=['True Not Survival', 'True Survival']
)


# In[ ]:



