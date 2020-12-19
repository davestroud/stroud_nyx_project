#%%
# Yes i used pandas.  Don't hate me because i'm beautiful
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds


mov_df = pd.read_csv("..\data\movies.csv", 
					header = 0,
					dtype={'movieId':'int32', 'title':'str', 'genres':'str'})
rate_df = pd.read_csv("..\data\mov_ratings.csv", 
					header = 0,
					dtype={'userId':'int32', 'movieId':'int32', 'rating':'float64'})
rate_df['timestamp'] = pd.to_datetime(rate_df.timestamp) #I like time stamps.  

# Saving for later when i want to group on decades. 
# rate_df['year'] = pd.DatetimeIndex(small_rate_df['timestamp']).year

#Poking around the movie df
# mov_df.describe()
# mov_df.columns
# mov_df.isnull().sum()

#How many movies are there?
print("How many movies were there\n\n", len(mov_df['movieId'].unique()))


# rate_df.describe()
# rate_df.columns
# rate_df.isnull().sum()
#Yay no nulls

# How many user rated a movie?
print(len(rate_df['userId'].unique()))
# Count how many reviews per user


# rate_df['year'] = pd.DatetimeIndex(small_rate_df['timestamp']).year
#%%

#For now lets subset this monster so it doesn't take 9 years to load in memory. 

small_rate_df = rate_df.iloc[:2000, :]

# Lets look at ratings for shits and giggles. 

mov_rating = rate_df.groupby(['movieId'], as_index=False)
avg_rating = mov_rating.agg({'rating':'mean'})
avg_rating.sort_values('rating', ascending=False)
# print(avg_rating.head(20))

#%%


# Alright enough EDA crap.  Lets try a basic SVD before we start slicing time windows
# into crazy little pieces and removing users like JS wants. 
# TODO SCIPY SVD MODEL

#First, lets pivot this df so the user is rows and the movid id is the columns. 
# Looking at a smaller subset first as doing the whole thing breaks my comp
 
small_rate_df = rate_df.iloc[:20000, :]
small_rate_pivot = small_rate_df.pivot(index='userId', 
										columns='movieId', 
										values='rating').fillna(0)


rating_matrix = small_rate_pivot.to_numpy()    #make it a matrix
rating_mean = np.mean(rating_matrix, axis=1)  #Take the mean
rate_demeaned = rating_matrix - rating_mean.reshape(-1, 1) # subtract mean off matrix

#SVD magic here
U, sigma, Vt = svds(rate_demeaned, k=50)
#Documentation told me so
sigma = np.diag(sigma)

predictions_mat = np.dot(np.dot(U, sigma), Vt) + rating_mean.reshape(-1, 1)

#Back to a dataframe. 
results_df = pd.DataFrame(predictions_mat, columns=small_rate_pivot.columns)

# %%

# Expected value for each user
# Go through original rating matrix to predictions
# when i calc a value of 5, what is it in expected predictions. 


# David idea
# Don't include someone who hasn't made a review in 25 years. 

# %%
# %%
rate_df.head(10)
