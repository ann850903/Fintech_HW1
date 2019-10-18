import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

training_filename = 'train.csv'
hidden_test_set_filename = 'test_no_G3.csv'
training_set_ratio = 0.8
alpha = 1
lambDa = 1

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

### training file
df = pd.read_csv(training_filename)
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

result = pd.concat([school, sex, age, famsize, studytime, failures, activities, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences], axis=1)
x = np.array(result)
y = np.array(G3)

training_set_num = int(np.size(x, 0)*training_set_ratio)

## training set (unnormalized)
x_train = x[0:training_set_num-1]
y_train = y[0:training_set_num-1]

## test set (unnormalized)
x_test = x[training_set_num:]
y_test = y[training_set_num:]

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
w_linear = np.dot(np.linalg.pinv(x_train), y_train)
y_test_predict_linear = np.dot(x_test, w_linear)
rmse_linear = rmse(y_test_predict_linear, y_test)
print("rmse_linear: ", rmse_linear)

## regularized linear regression model without the bias term
x_transpose = x_train.transpose()
Identity = np.identity(len(x_train[1,:]))
w_regularized = np.dot(np.linalg.inv(np.add(np.dot(x_transpose,x_train),lambDa*Identity)),np.dot(x_transpose,y_train))
y_test_predict_regularized = np.dot(x_test, w_regularized)
rmse_regularized = rmse(y_test_predict_regularized, y_test)
print("rmse_regularized: ", rmse_regularized)

## regularized linear regression model with the bias term
x_transpose_biased = x_train_biased.transpose()
Identity_biased = np.identity(len(x_train_biased[1,:]))
Identity_biased[0,0] = 0
w_regularized_biased = np.dot(np.linalg.inv(np.add(np.dot(x_transpose_biased,x_train_biased),lambDa*Identity_biased)),np.dot(x_transpose_biased,y_train))
y_test_predict_regularized_biased = np.dot(x_test_biased, w_regularized_biased)
rmse_regularized_biased = rmse(y_test_predict_regularized_biased, y_test)
print("rmse_regularized_biased: ", rmse_regularized_biased)

## Bayesian linear regression model with the bias term
w_Bayesian_biased = np.dot(np.linalg.inv(np.add(np.dot(x_transpose_biased,x_train_biased),alpha*Identity_biased)),np.dot(x_transpose_biased,y_train))
y_test_predict_Bayesian_biased = np.dot(x_test_biased, w_Bayesian_biased)
rmse_Bayesian_biased = rmse(y_test_predict_Bayesian_biased, y_test)
print("rmse_Bayesian_biased: ", alpha, rmse_Bayesian_biased)

### plot figure of results
## set title
plt.title('Regression result comparison')

## set y-label
plt.ylabel('Values')

## set x-label
plt.xlabel('Sample index')

## plot G3
Ground_Truth, = plt.plot(y_test)
Linear_Regression, = plt.plot(y_test_predict_linear)
Linear_Regression_reg, = plt.plot(y_test_predict_regularized)
Linear_Regression_reg_b, = plt.plot(y_test_predict_regularized_biased)
Bayesian_Linear_Regression, = plt.plot(y_test_predict_Bayesian_biased)

## set legend
plt.legend([Ground_Truth, Linear_Regression, Linear_Regression_reg, Linear_Regression_reg_b, Bayesian_Linear_Regression], \
['Ground Truth', '(' + '%.2f' % rmse_linear + ') Linear Regression', '(' + '%.2f' % rmse_regularized + ') Linear Regression (reg)', \
'(' + '%.2f' % rmse_regularized_biased + ') Linear Regression (r/b)', '(' + '%.2f' % rmse_Bayesian_biased + ') Bayesian Linear Regression'])

plt.show()

### hidden test set file
df = pd.read_csv(hidden_test_set_filename)
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

result = pd.concat([school, sex, age, famsize, studytime, failures, activities, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences], axis=1)
x = np.array(result)

## normalization
x_hidden = (x - x_train_mean) / x_train_std

## add bias
x_hidden_biased = np.column_stack((np.ones(len(x_hidden[:,1])),x_hidden))
y_hidden_predict = np.dot(x_hidden_biased, w_regularized_biased)

## unnormalization
y_hidden_predict = y_hidden_predict * y_train_std + y_train_mean

### output results
fo = open("r07943097_1.txt", "w")

ID = 1001
for g3 in y_hidden_predict:
    fo.write(str(ID) + "\t" + '%.1f' % g3 + "\n")
    ID += 1

fo.close()
