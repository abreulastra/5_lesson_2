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


train = pd.read_csv('https://raw.githubusercontent.com/nborwankar/LearnDataScience/master/datasets/samsung/samtrain.csv',index_col=0)
test = pd.read_csv('https://raw.githubusercontent.com/nborwankar/LearnDataScience/master/datasets/samsung/samtest.csv', index_col=0)
validate = pd.read_csv('https://raw.githubusercontent.com/nborwankar/LearnDataScience/master/datasets/samsung/samval.csv', index_col=0 )

features = ['tAccMean', 'tAccStd', 'tJerkMean', 'tJerkSD', 'tGyroMean','tGyroSD', 'tGyroJerkMean', 'tGyroJerkMagSD', 'fAccMean', 'fAccSD', 'fAccMeanFreq', 'fJerkMean', 'fJerkSD', 'fJerkMeanFreq', 'fGyroMean', 'fGyroSD', 'fGyroMeanFreq', 'fGyroJerkMean', 'fGyroJerkSD', 'fGyroJerkMeanFreq', 'fAccSkewness', 'fJerkSkewness', 'fGyroSkewness', 'fGyroJerkSkewness', 'fAccKurtosis', 'fJerkKurtosis', 'fGyroKurtosis', 'fGyroJerkKurtosis', 'angleAccGravity', 'angleJerkGravity', 'angleGyroGravity', 'angleGyroJerkGravity', 'angleXGravity', 'angleYGravity', 'angleZGravity']
X_train = train[features]
y_train = train.activity

X_test = test[features]
y_test = test.activity

X_validate = validate[features]
y_validate = validate.activity


# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 500, oob_score=True)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(X_train, y_train)

# Take the same decision trees and run it on the test data
y_test_pred = forest.predict(X_test)
y_validate_pred = forest.predict(X_validate)

print metrics.accuracy_score(y_test, y_test_pred)
print metrics.accuracy_score(y_validate, y_validate_pred)



#Full random forest
#changing the column names

df_rn = train
rn_dict = dict(zip(features,['x' + str(x) for x in range(len(features))]))
df_rn.rename(columns = rn_dict, inplace=True)

X_rn = df_rn[df_rn.columns[:35]]
y_rn = df_rn['activity']

