import numpy as np
import pandas as pd

filename = 'train.csv'
train_set_ratio = 0.8

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

train_set_num = int(np.size(x, 0)*train_set_ratio)

## training set (unnormalized)
x_train = x[0:train_set_num-1]
y_train = y[0:train_set_num-1]

## test set (unnormalized)
x_test = x[train_set_num:]
y_test = y[train_set_num:]

x_train_mean = np.mean(x_train, axis = 0)
x_train_std = np.std(x_train, axis = 0)
x_train_normalized = (x_train - x_train_mean) / x_train_std
x_test_normalized = (x_test - x_train_mean) / x_train_std

y_train_mean = np.mean(y_train, axis = 0)
y_train_std = np.std(y_train, axis = 0)
y_train_normalized = (y_train - y_train_mean) / y_train_std
y_test_normalized = (y_test - y_train_mean) / y_train_std

print(x_test_normalized)
print(y_test_normalized)
