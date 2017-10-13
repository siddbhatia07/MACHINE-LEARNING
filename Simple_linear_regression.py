# Data Preprocessing

# importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#splitting the dataset into Test and Training set

from sklearn.cross_validation import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

"""
#Feature Scalling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#fitting Simple Linear Regression to  the Training Set

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predincting the test set result

y_pred = regressor.predict(X_test)

#Visualising training set result

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train),  color = 'blue' )
plt.title('Salary vs Experience (training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train),  color = 'blue' )
plt.title('Salary vs Experience (test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()