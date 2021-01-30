# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 00:46:06 2021

@author: User
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support

data = pd.read_csv('voice.csv')
print(data)
print(data.groupby("label").count())
for col in data.columns:
    plt.hist(data.loc[data['label'] == 'female', col],label ="female")
    plt.hist(data.loc[data['label'] == 'male', col],label="male")
    plt.title(col)
    plt.xlabel("Feature magnitude")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.show()
#Encoding the label column. Female to 0 and male to 1
class_mapping = {label:idx for idx,label in enumerate(np.unique(data['label']))}
# Converting class labels from strings to integers
data['label'] = data['label'].map(class_mapping)

#Creating X,y and splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)

#Scaling the features

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


#Train support vector machine model

svm = SVC()
svm.fit(X_train_std, y_train)

print("Support Vector Machine")
print("Accuracy on training set: {:.3f}".format(svm.score(X_train_std, y_train)))
print("Accuracy on test set: {:.3f}".format(svm.score(X_test_std, y_test)))

y_pred_sm = svm.predict(X_test_std)
print("Predicted value: ",y_pred_sm)

precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred_sm, average='micro')
print("Precision, Recall and fscore:",precision, recall, fscore,)

#Read the file which got generated using our voice samples and using code written in R.

data_new = pd.read_csv("voiceSamples.csv")
data_new.head()


#Creating X and Y
X1, y1 = data_new.iloc[:, :-1].values, data_new.iloc[:, -1].values
y1

#standardizing the features
stdsc = StandardScaler()
X1_std = stdsc.fit_transform(X1)

#Predicting the target variable SVM
y1_pred_svm = svm.predict(X1_std)
print("SVM: ",y1_pred_svm)