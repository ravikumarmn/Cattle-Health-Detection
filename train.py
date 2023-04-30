import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import config
import joblib



# Load the dataset
dataframe = pd.read_csv('dataset/dataset_final.csv')
y = dataframe[config.TARGET]
X = dataframe[config.FEATURES]

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# SVM
svm = SVC()
svm.fit(X, y)
svm_y_pred = svm.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=8)
rf.fit(X, y)
rf_y_pred = rf.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_y_pred)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
joblib.dump(rf, 'random_forest_model.joblib')
rf_cm = confusion_matrix(y_test, rf_y_pred)
sns.heatmap(rf_cm, annot=True, fmt='g')

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Random Forest Confusion Matrix')
plt.savefig('webapp/static/rf_confusion_matrix.png', dpi=300, bbox_inches='tight')



svm_report = classification_report(y_test, svm_y_pred, zero_division=1)
rf_report = classification_report(y_test, rf_y_pred, zero_division=1)
