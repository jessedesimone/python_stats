#!/usr/local/bin/python3.9
"""
Copyright (C) 2021 Jesse DeSimone, Ph.D.

Some source code: https://www.kaggle.com/yogidsba/diabetes-prediction-eda-model

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
import numpy as np

# Read data
data = pd.read_csv("/Users/jessedesimone/DeSimone_Github/python_stats/Datasets/diabetes.csv")
data.info()

# Define predictors and outcome
X = data.drop('Outcome', axis=1)
y = data['Outcome']

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

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 4)
X_train.shape, y_train.shape
X_test.shape, y_test.shape

# Create and fit StandardScaler instance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaler = scaler.fit(X_train)

# Scaling data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled,columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled,columns=X_test.columns)

X_train_scaled_df.index=np.arange(len(X_train_scaled_df))
X_test_scaled_df.index=np.arange(len(X_test_scaled_df))
y_train.index=np.arange(len(y_train))
y_test.index=np.arange(len(y_test))

# Function to create confusion matrix
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,plot_confusion_matrix
def make_confusion_matrix(y_actual, y_predict, title):
    fig, ax = plt.subplots(1, 1)
    cm = confusion_matrix(y_actual, y_predict, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No", "Yes"])
    disp.plot(cmap='Reds', colorbar=True, ax=ax)
    ax.set_title(title)
    plt.tick_params(axis=u'both', which=u'both', length=0)
    plt.grid(b=None, axis='both', which='both', visible=False)
    plt.show()

from sklearn.metrics import  classification_report, accuracy_score, precision_score, recall_score,f1_score
# Function to calculate different metric scores of the model - Accuracy, Recall and Precision
def get_metrics_score(model, X_train_df, X_test_df, y_train_pass, y_test_pass, flag=True):
    '''
    model : classifier to predict values of X
    '''
    # defining an empty list to store train and test results
    score_list = []
    pred_train = model.predict(X_train_df)
    pred_test = model.predict(X_test_df)
    pred_train = np.round(pred_train)
    pred_test = np.round(pred_test)
    train_acc = accuracy_score(y_train_pass, pred_train)
    test_acc = accuracy_score(y_test_pass, pred_test)
    train_recall = recall_score(y_train_pass, pred_train)
    test_recall = recall_score(y_test_pass, pred_test)
    train_precision = precision_score(y_train_pass, pred_train)
    test_precision = precision_score(y_test_pass, pred_test)
    train_f1 = f1_score(y_train_pass, pred_train)
    test_f1 = f1_score(y_test_pass, pred_test)
    score_list.extend(
        (train_acc, test_acc, train_recall, test_recall, train_precision, test_precision, train_f1, test_f1))

    if flag == True:
        print("\x1b[0;30;47m \033[1mMODEL PERFORMANCE\x1b[0m")
        print("\x1b[0;30;47m \033[1mAccuracy   : Train:\x1b[0m", round(train_acc, 3),
              "\x1b[0;30;47m \033[1mTest:\x1b[0m ", round(test_acc, 3))
        print("\x1b[0;30;47m \033[1mRecall     : Train:\x1b[0m", round(train_recall, 3),
              "\x1b[0;30;47m \033[1mTest:\x1b[0m", round(test_recall, 3))

        print("\x1b[0;30;47m \033[1mPrecision  : Train:\x1b[0m", round(train_precision, 3),
              "\x1b[0;30;47m \033[1mTest:\x1b[0m ", round(test_precision, 3))
        print("\x1b[0;30;47m \033[1mF1         : Train:\x1b[0m", round(train_f1, 3),
              "\x1b[0;30;47m \033[1mTest:\x1b[0m", round(test_f1, 3))
        make_confusion_matrix(y_train_pass, pred_train, "Confusion Matrix for Train")
        make_confusion_matrix(y_test_pass, pred_test, "Confusion Matrix for Test")
    return score_list  # returning the list with train and test scores


# Build logistic regression model
from sklearn.linear_model import LogisticRegression
lr_mod = LogisticRegression()
lr_mod.fit(X_train_scaled_df, y_train)
score_list_dt=get_metrics_score(lr_mod,X_train_scaled_df,X_test_scaled_df,y_train,y_test)

# Cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lr_mod, X_train_scaled_df, y_train, cv=5)
print('Cross-Validation Accuracy Scores', scores)
scores = pd.Series(scores)
min_cv = scores.min()
max_cv = scores.max()
mean_cv = scores.mean()
print('Mean cross-validation score: {:.2f}'.format(mean_cv))

# Plot receiver operating characteristic (ROC) Area-under-the-curve (AUC)
from sklearn.metrics import roc_curve, auc
y_pred = lr_mod.decision_function(X_test)
logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred)
auc_logistic = auc(logistic_fpr, logistic_tpr)

plt.figure(figsize=(5, 5), dpi=100)
plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic (auc = %0.3f)' % auc_logistic)
plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')
plt.legend()
plt.show()