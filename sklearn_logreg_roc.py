#!/usr/local/bin/python3.9
"""
Copyright (C) 2021 Jesse DeSimone, Ph.D.

Change Log
=============
0.0.1 (2021-10-04)
-------------
Initial commit

"""
# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
data = pd.read_csv("/Users/jessedesimone/DeSimone_Github/python_stats/Datasets/diabetes.csv")
data.info()

# Define predictors and outcome
X = data.iloc[:,0:-1].values
y = data.iloc[:,-1].values

'''
Here, could use some preliminary correlation analysis to determine
most relevent features and remove some data with little to no
predictive value
'''

# Exemplar exploratory data analysis
X.shape, y.shape        #shape

sns.countplot(x='Outcome', data=data)
plt.xlabel('Disease (0 = No Diabetes)')
data.groupby('Outcome').mean()

pd.crosstab(data.Age,data.Outcome).plot(kind="bar",figsize=(10,6))
plt.title('Age grouped by Outcome')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.scatter(x=data.Glucose[data.Outcome==1], y=data.BMI[(data.Outcome==1)], c="red")
plt.scatter(x=data.Glucose[data.Outcome==0], y=data.BMI[(data.Outcome==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Blood Glucose")
plt.ylabel("Body Mass Index")
plt.show()

def dist_box(data):
 # function plots a combined graph for univariate analysis of continous variable
 #to check spread, central tendency , dispersion and outliers
    Name=data.name.upper()
    fig,(ax_box,ax_dis)  =plt.subplots(nrows=2,sharex=True,gridspec_kw = {"height_ratios": (.25, .75)},figsize=(8, 5))
    mean=data.mean()
    median=data.median()
    mode=data.mode().tolist()[0]
    sns.set_theme(style="white")
    sns.set_palette(sns.color_palette("Set1", 8))
    fig.suptitle("SPREAD OF DATA FOR "+ Name  , fontsize=18, fontweight='bold')
    sns.boxplot(x=data,showmeans=True, orient='h',ax=ax_box)
    ax_box.set(xlabel='')
     # just trying to make visualisation better. This will set background to white
    sns.despine(top=True,right=True,left=True) # to remove side line from graph
    sns.set_palette(sns.color_palette("Set1", 8))
    sns.distplot(data,kde=False,ax=ax_dis)
    ax_dis.axvline(mean, color='r', linestyle='--',linewidth=2)
    ax_dis.axvline(median, color='g', linestyle='-',linewidth=2)
    ax_dis.axvline(mode, color='y', linestyle='-',linewidth=2)
    plt.legend({'Mean':mean,'Median':median,'Mode':mode})

list_col = data.select_dtypes(include='number').columns.to_list()
for i in range(len(list_col)):
    dist_box(data[list_col[i]])

# Build logistic regression model

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 4)
X_train.shape, y_train.shape
X_test.shape, y_test.shape

# Use standard scaler to normalize the data across predictors
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()    #create StandardScaler instance
X_train = sc_X.fit_transform(X_train)     #fit StandardScaler
X_test = sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr_mod = LogisticRegression()
lr_mod.fit(X_train, y_train)
y_pred = lr_mod.predict(X_test)
y_pred_logistic = lr_mod.decision_function(X_test)  #use this option when plotting ROC

from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

# Accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# Cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lr_mod, X_train, y_train, cv=5)
print('Cross-Validation Accuracy Scores', scores)
scores = pd.Series(scores)
min_cv = scores.min()
max_cv = scores.max()
mean_cv = scores.mean()
print('Mean cross-validation score: {:.2f}'.format(mean_cv))

# Plot receiver operating characteristic (ROC) Area-under-the-curve (AUC)
from sklearn.metrics import roc_curve, auc
logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred_logistic)
auc_logistic = auc(logistic_fpr, logistic_tpr)

plt.figure(figsize=(5, 5), dpi=100)
plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic (auc = %0.3f)' % auc_logistic)
plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')
plt.legend()
plt.show()