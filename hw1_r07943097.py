import numpy as np
import pandas as pd
#from sklearn.linear_model import Ridge

filename = 'train.csv'
training_set_ratio = 0.8
alpha = 1

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

df = pd.read_csv(filename)
school = pd.get_dummies(df.school)
sex = pd.get_dummies(df.sex)
age = df.age
famsize = pd.get_dummies(df.famsize)
studytime = df.studytime
failures = df.failures
activities = pd.get_dummies(df.activities)
higher = pd.get_dummies(df.higher)
internet = pd.get_dummies(df.internet)
romantic = pd.get_dummies(df.romantic)
famrel = df.famrel
freetime = df.freetime
goout = df.goout
Dalc = df.Dalc
Walc = df.Walc
health = df.health
absences = df.absences
G3 = df.G3

res = pd.concat([school, sex, age, famsize, studytime, failures, activities, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences], axis=1)
x = np.array(res)
y = np.array(G3)

train_set_num = int(np.size(x, 0)*training_set_ratio)

## training set (unnormalized)
x_train = x[0:train_set_num-1]
y_train = y[0:train_set_num-1]

## test set (unnormalized)
x_test = x[train_set_num:]
y_test = y[train_set_num:]

## calculate column-wise mean and standard-deviation
x_train_mean = np.mean(x_train, axis = 0)
x_train_std = np.std(x_train, axis = 0)
y_train_mean = np.mean(y_train, axis = 0)
y_train_std = np.std(y_train, axis = 0)

## normalization
x_train = (x_train - x_train_mean) / x_train_std
x_test = (x_test - x_train_mean) / x_train_std
y_train = (y_train - y_train_mean) / y_train_std
y_test = (y_test - y_train_mean) / y_train_std

## add bias
x_train_biased = np.column_stack((np.ones(len(y_train)),x_train))
x_test_biased = np.column_stack((np.ones(len(y_test)),x_test))

## linear regression model without the bias term
w_unbiased = np.dot(np.linalg.pinv(x_train), y_train)
y_test_predict_linear = np.dot(x_test, w_unbiased)
#print("RMSE: ", rmse(y_test_predict_linear, y_test))

## Bayesian linear regression with the bias term
Xtranspose = x_train_biased.transpose()
Identity = np.identity(len(x_train_biased[1,:]))
Identity[0,0] = 0
w_Bayesian = np.dot(np.linalg.inv(np.add(np.dot(Xtranspose,x_train_biased),alpha*Identity)),np.dot(Xtranspose,y_train))
y_test_predict_Bayesian = np.dot(x_test_biased, w_Bayesian)
#print("RMSE: ", rmse(y_test_predict_Bayesian, y_test))
