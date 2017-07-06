# -*- coding: utf-8 -*-
"""
Created on Tue Jun 06 11:02:29 2017

@author: Aishu
"""
#packges to import 
import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt 

from scipy import stats 

import seaborn as sns 

# Load training set data
train = pd.read_csv("train.csv")

#Looking at the data 
#Null Values 
train.isnull().sum()

#new column person classifying based on age as male, female and children 
def male_female_child(passenger):
    age,sex = passenger
    if age< 16:
        return 'child' 
    else:
        return sex
        
train['person'] = train[['Age','Sex']].apply(male_female_child,axis=1)

#Visualization 

sns.factorplot('person',data = train,kind ="count")
sns.factorplot("Pclass",data =train,kind ="count",hue ="person")
train['Age'].hist(bins = 70)
print train['Age'].mean()
print train['person'].value_counts()
sns.factorplot("Embarked",data = train,kind="count")
sns.factorplot("SibSp",data = train,kind="count")
sns.factorplot("Parch",data = train,kind="count")

##Kde plot1
fig = sns.FacetGrid(train,hue ='Sex',aspect = 4)
fig.map(sns.kdeplot,'Age',shade = True)

oldest = train['Age'].max()
fig.set(xlim = (0,oldest,))
fig.add_legend()
##kde plot 2 
fig = sns.FacetGrid(train,hue ='person',aspect = 4)
fig.map(sns.kdeplot,'Age',shade = True)

oldest = train['Age'].max()
fig.set(xlim = (0,oldest,))
fig.add_legend()

##kde plot 3 
fig = sns.FacetGrid(train,hue ='Pclass',aspect = 4)
fig.map(sns.kdeplot,'Age',shade = True)

oldest = train['Age'].max()
fig.set(xlim = (0,oldest,))
fig.add_legend()

#Cabin analysis

deck = train['Cabin'].dropna()
levels =[]
for level in deck:
    levels.append(level[0])

cabin_df = pd.DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.factorplot("Cabin",data = cabin_df,kind = "count",palette = "winter")

cabin_df = cabin_df[cabin_df.Cabin != 'T']
sns.factorplot("Cabin",data = cabin_df,kind = "count",palette = "summer")

#Pclass on Survived

sns.factorplot("Survived",col = "Pclass",data = train[train.Pclass.notnull()],kind = "count",hue = "person")

def age_intervals(age):
    if age < 16:
        return 'Under 16'
    elif age < 25:
        return '16-25'
    elif age < 35:
        return '25-35'
    elif age < 45:
        return '35-45'
    elif age < 55:
        return '45-55'
    elif age < 65:
        return '55-65'
    else:
        return 'Above 65'
train['Age_intervals'] = train['Age'].apply(age_intervals)
train.head()
sns.factorplot("Survived",col = "Age_intervals",data = train[train.Age_intervals.notnull()],
               kind = "count",size=2.5, aspect=.8,hue ="Sex")

#Siblings and parents analysis

train["Alone"] = train['SibSp']+train['Parch']
train['Alone'].loc[train['Alone']>0] ="With Family"
train['Alone'].loc[train['Alone']==0] ="Alone"

##
sns.factorplot("Alone",data = train,kind = "count")
sns.factorplot("Pclass","Survived",hue = "person",data = train)
sns.lmplot("Age","Survived",data =train,hue = "Pclass")

generation = [10,20,40,60,80]
sns.lmplot("Age","Survived",data =train,hue = "Sex",x_bins = generation)
sns.lmplot("Age","Survived",data =train,hue = "Sex")

sns.factorplot("Survived",col = "Alone",data = train, kind = "count",hue = "Pclass")
sns.lmplot("Age","Survived",data =train,hue = "Pclass",x_bins = generation)
