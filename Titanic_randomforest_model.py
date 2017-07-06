# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 14:17:29 2017

@author: Aishu
"""
#importing packages
import numpy as np 
import pandas as pd

train = pd.read_csv('train.csv',index_col = 0)

def male_female_child(passenger):
    age,sex = passenger
    if age< 16:
        return 'child' 
    else:
        return sex
train['person'] = train[['Age','Sex']].apply(male_female_child,axis=1)

#creating dummies for categorical data

dummies = pd.get_dummies(train["person"])
dummies.head()

X = train[["Pclass","SibSp","Parch"]]
Y = train["Survived"]

X = pd.concat([X,dummies],axis = 1)
X.head()

#fitting the model using random forest
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X,Y)

#reading the test data 
test = pd.read_csv('test.csv',index_col = 0)
test.head()

test['person'] = test[['Age','Sex']].apply(male_female_child,axis=1)
dummies1 = pd.get_dummies(test["person"])
dummies1.head()

X_test = test[["Pclass","SibSp","Parch"]]
X_test = pd.concat([X_test,dummies1],axis = 1)
X_test.head()

pred = model.predict(X_test)
test["Survived"] = pred

test.to_csv("Test_data_prediction.csv")
model.score(X,Y)
