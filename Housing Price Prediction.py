
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 19:11:26 2022

@author: Benjamin Sin
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

raw_df = pd.read_csv('kc_house_data.csv')

# Exploratory Analysis on training data set
print(raw_df.head(10))
print(raw_df.describe())
print(raw_df.info())
print(raw_df.nunique())
print(raw_df.isna().sum())

# Correlation plot to understand the level of correlation between each variable and target
correlation = raw_df.corr()

# finding for outliers in pricing data
plt.figure(figsize=(16,5))
sns.histplot(raw_df.price, kde=True)

plt.show()

# We will use IQR to filter outliers in pricing data
Q1,Q3 = np.percentile(sorted(raw_df.price) , [25,75])
IQR = Q3-Q1
lower_range = Q1-(1.5 * IQR)
upper_range = Q3+(1.5 * IQR)

raw_df = raw_df[(raw_df.price >= lower_range)&(raw_df.price <= upper_range)]

# New plot of price distribution
plt.figure(figsize=(16,5))
sns.histplot(raw_df.price, kde=True)

plt.show()

# Plot of heat map to visualize correlation
plt.figure(figsize=(16,12))
plt.title('Correlation of Variables in kc house DataSet')
ax = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=30)           
plt.show()

# Dropping data (columns) that are not useful
y = raw_df.price
X = raw_df.drop(['id','price','date','sqft_living15','sqft_above'], axis=1)


# Splitting the data into 70% training data and 30% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Initializing the Regression models
LR = LinearRegression()
DT = DecisionTreeRegressor()
SVR = SVR()
ridge = Ridge()
RF = RandomForestRegressor()


# Comparing Cross Validation score for the different models
kf = KFold(n_splits=5, shuffle=True, random_state= 42)

cv_results_LR = cross_val_score(LR,X_train,y_train,cv=kf)
cv_results_DT = cross_val_score(DT,X_train,y_train,cv=kf)
cv_results_SVR = cross_val_score(SVR,X_train,y_train,cv=kf)
cv_results_ridge = cross_val_score(ridge,X_train,y_train,cv=kf)
cv_results_RF = cross_val_score(RF,X_train,y_train,cv=kf)

print("LinearRegression Cross Validation Score: {}".format(np.mean(cv_results_LR)))
print("DecisionTree Regressor Cross Validation Score: {}".format(np.mean(cv_results_DT)))
print("SVR Cross Validation Score: {}".format(np.mean(cv_results_SVR)))
print("Ridge Cross Validation Score: {}".format(np.mean(cv_results_ridge)))
print("RandomForest Regressor Cross Validation Score: {}".format(np.mean(cv_results_RF)))
print('\n')

# fitting the models (exclude SVR, Linear Regression and Ridge Regression due to lower scores) with training data
DT.fit(X_train,y_train)
RF.fit(X_train,y_train)


# predicting the y value for the top 2 performing model
predict_DT = DT.predict(X_test)
predict_RF = RF.predict(X_test)


# Scoring the different models using the training data and test data to check for over-fit or under-fit
print("DecisionTree Regressor score on Training data: {}".format(DT.score(X_train,y_train)))
print("DecisionTree Regressor score on Test data: {}".format(DT.score(X_test,y_test)))
print('\n')

print("RandomForest Regressor score on Training data: {}".format(RF.score(X_train,y_train)))
print("RandomForest Regressor score on Test data: {}".format(RF.score(X_test,y_test)))
print('\n')


# MSE for the different models
print("MSE for Decision Tree Regression: {}".format(mse(y_test,predict_DT)))
print("MSE for Random Forest Regression: {}".format(mse(y_test,predict_RF)))


# Tuning DecisionTreeRegressor
parameters_DT = {'max_depth': np.arange(2,10),
              'min_samples_split' : np.arange(2,20,2),
              'min_samples_leaf' : np.arange(1,21,2)} 

DT_tuning = RandomizedSearchCV(DT,parameters_DT,cv=kf)
DT_tuning.fit(X_train,y_train)
print('\n')
print(DT_tuning.best_params_, DT_tuning.best_score_)
print('\n')
# Checking for under-fitting or over-fitting
# Also to check if test score has improved after hyper parameter tuning
print("Tuned Decision Tree Score on training data: {}".format(DT_tuning.score(X_train,y_train)))
print("Tuned Decision Tree Score on test data: {}".format(DT_tuning.score(X_test,y_test)))


# Tuning RandomForest Regressor
parameters_RF = {'bootstrap': [True, False],
                     'max_depth': [10, 20, 30, 40, 50],
                     'max_features': ['auto', 'sqrt'],
                     'min_samples_leaf': [1, 2, 4],
                     'min_samples_split': [2, 6, 10],
                     'n_estimators': [200, 400, 600, 800, 1000]}


RF_tuning = RandomizedSearchCV(RF,parameters_RF,cv=kf)
RF_tuning.fit(X_train,y_train)

print('\n')
print(RF_tuning.best_params_, RF_tuning.best_score_)
print('\n')
# Checking for under-fitting or over-fitting
# Also to check if test score has improved after hyper parameter tuning
print("Tuned Random Forest Regressor Score on training data: {}".format(RF_tuning.score(X_train,y_train)))
print("Tuned Random Forest Regressor Score on test data: {}".format(RF_tuning.score(X_test,y_test)))
print('\n')
final_y_pred = RF_tuning.predict(X_test)
print("MSE for Tuned Decision Tree Regression: {}".format(mse(y_test,final_y_pred)))
print("Tuned Random Forest Regressor R^2 Score: {}".format(r2_score(y_test,final_y_pred)))



