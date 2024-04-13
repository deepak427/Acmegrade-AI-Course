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

