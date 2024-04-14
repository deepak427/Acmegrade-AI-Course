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

