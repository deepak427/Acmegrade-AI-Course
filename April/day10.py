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
y= exercise_df.filter(["kind"])

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

# Clustering

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dataset= pd.read_csv("April/data/Mall_Customers.csv")
print(dataset.head())

# Plotting

# sns.displot(dataset["Annual Income (k$)"], kde= False, bins=50, color="blue")
# plt.show()

# sns.displot(dataset["Spending Score (1-100)"], kde= False, bins=50, color="red")
# plt.show()

# sns.regplot(x= "Annual Income (k$)", y= "Spending Score (1-100)", data=dataset)
# plt.show()

# sns.regplot(x= "Age", y= "Spending Score (1-100)", data=dataset)
# plt.show()

dataset= dataset.filter(["Spending Score (1-100)", "Annual Income (k$)"], axis=1)
print(dataset.head())

km_model= KMeans(n_clusters=4)
km_model.fit(dataset)

print(km_model.cluster_centers_)
print(km_model.labels_)

plt.scatter(dataset.values[:,0], dataset.values[:,1], c=km_model.labels_, cmap='rainbow')
plt.scatter(km_model.cluster_centers_[:,0], km_model.cluster_centers_[:,1], s=100, c="black")
plt.show()

# Elbow method

loss=[]

for i in range(1, 11):
    km=KMeans(n_clusters=i).fit(dataset)
    loss.append(km.inertia_)

plt.plot(range(1,11), loss)
plt.title("Finding optimal number of clusters via elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("loss")
plt.show()

km_model= KMeans(n_clusters=5)
km_model.fit(dataset)

print(km_model.cluster_centers_)
print(km_model.labels_)

plt.scatter(dataset.values[:,0], dataset.values[:,1], c=km_model.labels_, cmap='rainbow')
plt.scatter(km_model.cluster_centers_[:,0], km_model.cluster_centers_[:,1], s=100, c="black")
plt.show()

