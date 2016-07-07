# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 22:49:53 2016

@author: AbreuLastra_Work
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.metrics import confusion_matrix


df = pd.read_csv('https://raw.githubusercontent.com/nborwankar/LearnDataScience/master/datasets/samsung/samtest.csv',index_col=0)


features = ['tAccMean', 'tAccStd', 'tJerkMean', 'tJerkSD', 'tGyroMean','tGyroSD', 'tGyroJerkMean', 'tGyroJerkMagSD', 'fAccMean', 'fAccSD', 'fAccMeanFreq', 'fJerkMean', 'fJerkSD', 'fJerkMeanFreq', 'fGyroMean', 'fGyroSD', 'fGyroMeanFreq', 'fGyroJerkMean', 'fGyroJerkSD', 'fGyroJerkMeanFreq', 'fAccSkewness', 'fJerkSkewness', 'fGyroSkewness', 'fGyroJerkSkewness', 'fAccKurtosis', 'fJerkKurtosis', 'fGyroKurtosis', 'fGyroJerkKurtosis', 'angleAccGravity', 'angleJerkGravity', 'angleGyroGravity', 'angleGyroJerkGravity', 'angleXGravity', 'angleYGravity', 'angleZGravity']
X = df[features]
y = df.activity

print(df.shape)

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 500)

###First we run a model spliting the data set in two

# use train/test split with different random_state values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)


# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(X_train, y_train)

# Take the same decision trees and run it on the test data
y_pred = forest.predict(X_test)

print metrics.accuracy_score(y_test, y_pred)

# Then we do cross validation
scores = cross_val_score(forest, X, y, cv=10, scoring='accuracy')
print scores
print scores.mean()
print forest.feature_importances_

#Full random forest
#changing the column names

df_rn = df
rn_dict = dict(zip(features,['x' + str(x) for x in range(len(features))]))
df_rn.rename(columns = rn_dict, inplace=True)

X_rn = df_rn[df_rn.columns[:35]]
y_rn = df_rn['activity']


#Spliting the dataset 
"""
training_rn = df_rn[df_rn['subject'] >= 27]
test_rn = df_rn[df_rn['subject'] <=6 ]



X_test_rn = test_rn[test_rn.columns[:35]]
y_test_rn = test_rn['activity']
"""

# Fit a random forest with 50 estimators
forest_rn = RandomForestClassifier(n_estimators = 50)
scores_rn = cross_val_score(forest, X_rn, y_rn, cv=10, scoring='accuracy')
print(scores_rn)
print scores.mean()
print 


y_rn_pred= forest.predict(X_rn)

print metrics.accuracy_score(y_rn, y_rn_pred)



