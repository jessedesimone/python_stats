#!/usr/local/bin/python3.9
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

inpath = '<file path>'
df = pd.read_csv(inpath + '<infile>')

df.shape
df.info()
df.head()
df.describe()

space=df['sqft_living']
price=df['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

#Splitting the data into Train and Test
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1/3, random_state=0)

#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)
print('linear equation = ', regressor.coef_, 'x', regressor.intercept_)

#Predicting the prices
pred = regressor.predict(xtest)
#Create dataframe of predicted vs actual values
df2 = pd.DataFrame({'Actual': ytest, 'Predicted': pred})
df2

#Visualizing the training Test Results
plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

#Visualizing the training Test Results
plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#Predict new value
new_X = 4000
X = np.array(new_X).reshape(-1,1)
regressor.predict(X)
