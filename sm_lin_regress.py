#!/usr/local/bin/python3.9

'''
Code derived from Kaushik Katari (https://towardsdatascience.com/simple-linear-regression-model-using-python-machine-learning-eab7924d18b4)
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import r2_score

# Read the given CSV file, and view some sample records
inpath = '<path/to/file>'
df = pd.read_csv(inpath + 'Company_data.csv')
df.info()
df.describe()

# Visualize the data wrt to Sales
sns.pairplot(df, x_vars=['TV', 'Radio','Newspaper'],
             y_vars='Sales', size=4, aspect=1, kind='scatter')
plt.show()

# Visualize the data using heatmap
sns.heatmap(df.corr(), cmap="YlGnBu", annot = True)
plt.show()

# Create X and y
X = df['TV']
y = df['Sales']

# # Splitting the varaibles as training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size = 0.7,
                                                    test_size = 0.3,
                                                    random_state = 100)
X_train
y_train

# Building and training the model
# Adding a constant to get an intercept
X_train_sm = sm.add_constant(X_train)

# Fitting the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()

# Printing the parameters
lr.params

# Performing a summary to list out all the different parameters of the regression line fitted
lr.summary()

# Visualizing the regression line
plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()

# Residual analysis
# Predicting y_value using traingn data of X
y_train_pred = lr.predict(X_train_sm)

# Creating residuals from the y_train data and predicted y_data
res = (y_train - y_train_pred)

# Plotting the histogram using the residual values
fig = plt.figure()
sns.distplot(res, bins = 15)
plt.title('Error Terms', fontsize = 15)
plt.xlabel('y_train - y_train_pred', fontsize = 15)
plt.show()

# Looking for any patterns in the residuals
plt.scatter(X_train,res)
plt.show()

# Predictions on the Test data or Evaluating the model
# Adding a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Predicting the y values corresponding to X_test_sm
y_test_pred = lr.predict(X_test_sm)

# Printing the first 15 predicted values
y_test_pred

# Checking the R-squared value
r_squared = r2_score(y_test, y_test_pred)
r_squared

'''
Since the R² value on test data is within 5% of the R² value 
on training data, we can conclude that the model is pretty stable. 
Which means, what the model has learned on the training set can 
generalize on the unseen test set.
'''

# Visualize the line on the test set
plt.scatter(X_test, y_test)
plt.plot(X_test, y_test_pred, 'r')
plt.show()


