#!/usr/local/bin/python3.9
'''
Copyright (C) 2022 Jesse DeSimone, Ph.D.

TODO: fix optimal cutoff code
'''
#import packages
import warnings
import sys
sys.path.insert(0, '/Users/jessedesimone/DeSimone_Github/P01_MCI_DMRI/p01_mci_dmri') #configure system path
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from reader.reader import *
import reader.reader as reader

#configure directories
BASE_DIR = reader.directories.BASE_DIR
PROJ_DIR = reader.directories.PROJ_DIR
DATA_DIR = reader.directories.DATA_DIR
ANALYSIS_DIR = reader.directories.ANALYSIS_DIR
FUNCTION_DIR = reader.directories.FUNCTION_DIR
RUNNER_DIR = reader.directories.RUNNER_DIR
TEST_DIR = reader.directories.TEST_DIR
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)
OUT_DIR = reader.directories.OUT_DIR
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

#define functions
def make_confusion_matrix(y_actual, y_predict, title):
    '''confusion matrix function'''
    fig, ax = plt.subplots(1, 1)
    cm = confusion_matrix(y_actual, y_predict, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    disp.plot(cmap='Blues', colorbar=True, ax=ax)
    ax.set_title(title)
    plt.tick_params(axis=u'both', which=u'both', length=0)
    plt.grid(b=None, axis='both', which='both', visible=False)
    plt.show()

def get_metrics_score(model, X_train_df, X_test_df, y_train_pass, y_test_pass, flag=True):
    '''
    function to calculate different metric scores of the model - Accuracy, Recall and Precision
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

def dist_box(data):
    '''
    function plots a combined graph for univariate analysis of continous variable
    to check spread, central tendency , dispersion and outliers
    '''
    Name=data.name.upper()
    fig,(ax_box,ax_dis)  =plt.subplots(nrows=2,sharex=True,gridspec_kw = {"height_ratios": (.25, .75)},figsize=(8, 5))
    mean=data.mean()
    median=data.median()
    #mode=data.mode().tolist()[0]
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
    #ax_dis.axvline(mode, color='y', linestyle='-',linewidth=2)
    plt.legend({'Mean':mean,'Median':median})

#read data
excel_reader = reader.ExcelFileReader()
df = excel_reader.read_data(DATA_DIR + "youden.xlsx")
assert isinstance(df, pd.DataFrame)
pd.set_option('display.max_columns', None)

#Exploratory analysis
#plot amylpet counts
plt.figure()
sns.countplot(x=y, data=df)
plt.xlabel('Amyloid PET')
plt.show()

#correlation plot
corr=df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap='mako_r', annot=True)
plt.show()

#spread of continuous variables
#create list of continuous variables
list_col = df.select_dtypes(include='number').columns.to_list()
list_col.remove('amylpet')          #drop amylpet; categorical but treated by python as continuous
for i in range(len(list_col)):
    dist_box(df[list_col[i]])

#means grouped by amylpet
df.groupby(df['amylpet']).mean()

#scatter plot AB42_40 vs SUVR grouped by PET rating (+/-)
plt.figure(figsize=(5,3))
plt.scatter(x=df.suvr[df.amylpet==1], y=df.ab42_40[(df.amylpet==1)], c="red")
plt.scatter(x=df.suvr[df.amylpet==0], y=df.ab42_40[(df.amylpet==0)], c="blue")
plt.legend(["Amyloid PET+", "Amyloid PET-"])
plt.axhline(y = 0.16, color = 'black', linestyle = 'dashed')
plt.xlabel("SUVR")
plt.ylabel("Plasma AB42/40")
plt.show()

#Select data
'''
Option 1
Select highly correlated features (threshold=0.2)
'''
#cor_target = abs(corr["amylpet"])       #get absolute value of the correlation
#relevant_features = cor_target[cor_target>0.2]
#features_f=[index for index, value in relevant_features.iteritems()]
#features_f.remove('amylpet')   #remove outcome from list of features
#print(features_f)
#X=df[features_f]
#y=df['amylpet']

'''
Option 2
Manually select features
'''
X=df['ab42_40']
y=df['amylpet']

#get shape
X.shape, y.shape

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 4)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)

#create and fit StandardScaler instance
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)

#build logistic regression model
lr_mod = LogisticRegression()
lr_mod.fit(X_train_scaled, y_train)
score_list_dt=get_metrics_score(lr_mod,X_train_scaled,X_test_scaled,y_train,y_test)

#cross validation
scores = cross_val_score(lr_mod, X_train_scaled, y_train, cv=5)
print('Cross-Validation Accuracy Scores', scores)
scores = pd.Series(scores)
min_cv = scores.min()
max_cv = scores.max()
mean_cv = scores.mean()
print('Mean cross-validation score: {:.2f}'.format(mean_cv))

#plot ROC and calculate AUC
y_pred = lr_mod.decision_function(X_test_scaled)
logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred)
auc_logistic = auc(logistic_fpr, logistic_tpr)
plt.figure(figsize=(5, 5), dpi=100)
plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic (auc = %0.3f)' % auc_logistic)
plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')
plt.legend()
plt.show()

#optimal cutoff threshold
optimal_idx = np.argmax(logistic_tpr - logistic_fpr)
optimal_threshold = threshold[optimal_idx]
print("Threshold value is:", optimal_threshold)
