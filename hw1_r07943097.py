import numpy as np
import pandas as pd

filename = 'train.csv'

df = pd.read_csv(filename)
school = pd.get_dummies(df.school)
#print(school)
sex = pd.get_dummies(df.sex)
#print(sex)
age = df.age
#print(age)
famsize = pd.get_dummies(df.famsize)
#print(famsize)
studytime = df.studytime
#print(studytime)
failures = df.failures
#print(failures)
activities = pd.get_dummies(df.activities)
#print(activities)
higher = pd.get_dummies(df.higher)
#print(higher)
internet = pd.get_dummies(df.internet)
#print(internet)
romantic = pd.get_dummies(df.romantic)
#print(romantic)
famrel = df.famrel
#print(famrel)
freetime = df.freetime
#print(freetime)
goout = df.goout
#print(goout)
Dalc = df.Dalc
#print(Dalc)
Walc = df.Walc
#print(Walc)
health = df.health
#print(health)
absences = df.absences
#print(absences)
G3 = df.G3

res = pd.concat([school, sex, age, famsize, studytime, failures, activities, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences], axis=1)
#print(res)
x = np.array(res)
print(x)
#y = np.array(G3)
#print(y)


