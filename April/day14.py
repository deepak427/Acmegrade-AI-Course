# Movie Recommender system

# import dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

movie_ids_title=pd.read_csv("April/data/movies.csv")
movie_ids_rating=pd.read_csv("April/data/ratings.csv")

print(movie_ids_title.head())
print(movie_ids_rating.head())

# Preprocessing

movie_ids_title.drop(["genres"], inplace=True, axis=1)
movie_ids_rating.drop(["timestamp"], inplace=True, axis=1)

merged_movie_df=pd.merge(movie_ids_rating, movie_ids_title, on="movieId")

print(merged_movie_df.head())

print(merged_movie_df.groupby('title')['rating'].mean().sort_values(ascending=False).head())
print(merged_movie_df.groupby('title')['rating'].count().sort_values(ascending=False).head())

# dataframe that show the title, mean rating and the rating counts

movie_rating_mean_count = pd.DataFrame(columns=['rating_mean', 'rating_count'])
movie_rating_mean_count["rating_mean"]=merged_movie_df.groupby("title")["rating"].mean()
movie_rating_mean_count["rating_count"]=merged_movie_df.groupby("title")["rating"].count()

print(movie_rating_mean_count.head())

# visualization

plt.figure(figsize=(10,8))
sns.set_style("darkgrid")

movie_rating_mean_count["rating_mean"].hist(bins=30, color="purple")
# plt.show()

plt.figure(figsize=(10,8))
sns.set_style("darkgrid")

movie_rating_mean_count["rating_count"].hist(bins=33, color="purple")
# plt.show()

plt.figure(figsize=(10,8))
sns.set_style("darkgrid")

sns.regplot(x="rating_mean", y="rating_count", data=movie_rating_mean_count, color="brown")
# plt.show()

print(movie_rating_mean_count.sort_values("rating_count", ascending=False).head())

# Item based collaborating filtering

user_movie_rating_matrix = merged_movie_df.pivot_table(index="userId", columns="title", values="rating")
print(user_movie_rating_matrix)

pulp_fiction_rating = user_movie_rating_matrix["Pulp Fiction (1994)"]
pulp_fiction_corelations=pd.DataFrame(user_movie_rating_matrix.corrwith(pulp_fiction_rating), columns=["pf_corr"])
print(pulp_fiction_corelations.sort_values("pf_corr",ascending=False).head())

pulp_fiction_corelations=pulp_fiction_corelations.join(movie_rating_mean_count["rating_count"])
print(pulp_fiction_corelations.head())
pulp_fiction_corelations.dropna(inplace=True)
print(pulp_fiction_corelations.sort_values("pf_corr",ascending=False).head())

pulp_fiction_corelations_50=pulp_fiction_corelations[pulp_fiction_corelations['rating_count']>50]
print(pulp_fiction_corelations_50.sort_values("pf_corr", ascending=False).head())

all_movie_corelations = user_movie_rating_matrix.corr(method="pearson", min_periods=50)