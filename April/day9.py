import seaborn as sns
import numpy as np
import pandas as pd
# Regression problems in ML

# print(sns.get_dataset_names())

# Import dataset

tips_df=sns.load_dataset('tips')

print(tips_df.head())

X=tips_df.drop(["tip"], axis=1)
y=tips_df[["tip"]]

# ategorical Data Encoding

numerical= X.drop(["sex", "smoker", "day", "time"], axis=1)
categorical= X.filter(["sex", "smoker", "day", "time"])

print(categorical["day"].value_counts())

cat_numerical = pd.get_dummies(categorical)
print(cat_numerical.head())

X= pd.concat([numerical, cat_numerical], axis=1)
print(X.head())

# Divide dataset into train and test (optional: Validate)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.20, random_state=0)

# data scaling/ Normalization

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train= sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Linear Regression

from sklearn.linear_model import LinearRegression

lin_reg= LinearRegression()
regressor= lin_reg.fit(X_train, y_train)
y_pred= regressor.predict(X_test)

# Merices

# Mean absolute error | Mean squared error | Root mean squared error

from sklearn import metrics

print("Mean absolute error: " , metrics.mean_absolute_error(y_test, y_pred))
print("Mean squared error: " , metrics.mean_squared_error(y_test, y_pred))
print("Root mean squared error: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Random Forest

from sklearn.ensemble import RandomForestRegressor

rf_reg= RandomForestRegressor(random_state=42, n_estimators=500)
regressor= rf_reg.fit(X_train, y_train)
y_pred= regressor.predict(X_test)

from sklearn import metrics

print("Mean absolute error: " , metrics.mean_absolute_error(y_test, y_pred))
print("Mean squared error: " , metrics.mean_squared_error(y_test, y_pred))
print("Root mean squared error: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))