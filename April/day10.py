# Import dependencies
import seaborn as sns
import pandas as pd
import numpy as np

# Clasification problem in ML

exercise_df = sns.load_dataset("exercise")
print(exercise_df.head())

exercise_df = exercise_df.drop(["id", "Unnamed: 0"], axis=1)
print(exercise_df.head())

# Dividing data into 
X= exercise_df.drop(["kind"], axis=1)
y= exercise_df.filter("kind")

# Converting catrgorical data to numerical

numerical= X.filter(['pulse'])

categorical= X.filter(['diet', 'time'])
cat_numerical= pd.get_dummies(categorical)
print(cat_numerical.head())

X= pd.concat([numerical, cat_numerical], axis=1)
print(X.head())

# Splitting data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.20, random_state=0)

# Data Scaling/ Normalization

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train= sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Model 

from sklearn.linear_model import LogisticRegression
log_clf= LogisticRegression()
clasifier= log_clf.fit(X_train, y_train)
y_pred= clasifier.predict(X_test)

# Performance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Random forest classifier

from sklearn.ensemble import RandomForestClassifier

rf_clf= RandomForestClassifier(random_state=42, n_estimators=500)
clasifier= rf_clf.fit(X_train, y_train)
y_pred= clasifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))




