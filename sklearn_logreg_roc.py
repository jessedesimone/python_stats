#!/usr/local/bin/python3.9
"""
Copyright (C) 2021 Jesse DeSimone, Ph.D.

Sources:
https://www.kaggle.com/code/fareselmenshawii/logistic-regression
https://www.kaggle.com/yogidsba/diabetes-prediction-eda-model

"""
# Import packages
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score,f1_score
from sklearn.model_selection import cross_val_score

#define functions
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

# Function to calculate different metric scores of the model - Accuracy, Recall and Precision
def get_metrics_score(model, X_train, X_test, y_train, y_test, flag=True):
    '''
    model : classifier to predict values of X
    '''
    # defining an empty list to store train and test results
    score_list = []
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    pred_train = np.round(pred_train)
    pred_test = np.round(pred_test)
    train_acc = accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)
    train_recall = recall_score(y_train, pred_train)
    test_recall = recall_score(y_test, pred_test)
    train_precision = precision_score(y_train, pred_train)
    test_precision = precision_score(y_test, pred_test)
    train_f1 = f1_score(y_train, pred_train)
    test_f1 = f1_score(y_test, pred_test)
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
        make_confusion_matrix(y_train, pred_train, "Confusion Matrix for Train")
        make_confusion_matrix(y_test, pred_test, "Confusion Matrix for Test")
    return score_list  # returning the list with train and test scores

#read data
BASE_DIR='/Users/jessedesimone/DeSimone_Github/python_stats/'
DATA_DIR=os.path.join(BASE_DIR + 'Datasets/')
df = pd.read_csv(DATA_DIR + 'diabetes.csv')
df.info()
df.describe().T

#exploratory data analysis
#correlation plot
corr=df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap='mako_r', annot=True)
plt.show()

#scatter plot AB42_40 vs SUVR grouped by PET rating (+/-)
plt.scatter(x=df.Glucose[df.Outcome==1], y=df.BMI[(df.Outcome==1)], c="red")
plt.scatter(x=df.Glucose[df.Outcome==0], y=df.BMI[(df.Outcome==0)], c="blue")
plt.legend(["Diabetes+", "Diabetes-"])
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.show()

#select highly correlated features (threshold=0.2)
cor_target = abs(corr["Outcome"]) #get absolute value of the correlation
relevant_features = cor_target[cor_target>0.2]
features_f=[index for index, value in relevant_features.iteritems()]
features_f.remove('Outcome')   #remove outcome from list of features
features_f.remove('Pregnancies')   #remove pregnancies from list of features
print(features_f)

#define features and target
X=df[features_f]
y=df['Outcome']

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42) #split the  data into traing and validating

#scale data
'''scale features with continuous variables'''
scaler = StandardScaler() #create an instance of standard scaler
scaler.fit(X_train) # fit it to the training data

X_train = scaler.transform(X_train) #transform training data
X_test = scaler.transform(X_test) #transform validation data

#create model
model = LogisticRegression() #create logistic regression instance
model.fit(X_train, y_train)  # fit the model instance
predictions = model.predict(X_test) # calculate predictions

#evaluate model
#compute metrics for evaluation
accuracy = accuracy_score(y_test, predictions)
print(f'the model accuracy: {accuracy}')
score_list_dt=get_metrics_score(model,X_train,X_test,y_train,y_test)

from sklearn.metrics import classification_report
print (classification_report(y_test, predictions))
#cross validation
scores = cross_val_score(model, X_train, y_train, cv=5)
print('Cross-Validation Accuracy Scores', scores)
scores = pd.Series(scores)
min_cv = scores.min()
max_cv = scores.max()
mean_cv = scores.mean()
print('Mean cross-validation accuracy: {:.2f}'.format(mean_cv))

from sklearn.metrics import roc_curve, auc
y_pred = model.decision_function(X_test)
logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred)
auc_logistic = auc(logistic_fpr, logistic_tpr)
print('Area under the curve is: ', auc_logistic)
plt.figure(figsize=(5, 5), dpi=100)
plt.plot([0,1], [0,1], linestyle="--") # plot random curve
plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic (auc = %0.3f)' % auc_logistic)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')
plt.legend(loc='lower right')
plt.show()

#optimal cutoff threshold
optimal_idx = np.argmax(logistic_tpr - logistic_fpr)
optimal_threshold = threshold[optimal_idx]
print("Threshold value is:", optimal_threshold)