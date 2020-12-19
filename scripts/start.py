#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split


mov_df = pd.read_csv("..\data\movies.csv", header = 0)
rate_df = pd.read_csv("..\data\mov_ratings.csv", header = 0)

#Poking around the movie df
mov_df.describe()
mov_df.columns
mov_df.isnull().sum()

#How many movies are there?
print(len(mov_df['movieId'].unique()))


rate_df.describe()
rate_df.columns
rate_df.isnull().sum()
#Yay no nulls

# How many user rated a movie?
print(len(rate_df['userId'].unique()))


rate_df['timestamp'] = pd.to_datetime(rate_df.timestamp)
# rate_df['year'] = pd.DatetimeIndex(small_rate_df['timestamp']).year
#%%

small_rate_df = rate_df.iloc[:200, :]
small_rate_df['timestamp'] = pd.to_datetime(rate_df.timestamp)
small_rate_df['year'] = pd.DatetimeIndex(small_rate_df['timestamp']).year


#Next lets merge the ratings and movies dataframe. 
move_df.merge(rate_df, )

# Lets look at ratings for shits and giggles. 

mov_rating = rate_df.groupby(['movieId'], as_index=False)
avg_rating = mov_rating.agg({'rating':'mean'})
avg_rating.sort_values('rating', ascending=False)


