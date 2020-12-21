#%%
# Yes i used pandas.  Don't hate me because i'm beautiful
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
import timeit


# mov_df = pd.read_csv("..\data\movies.csv", 
# 					header = 0,
# 					
# type={'movieId':'int32', 'title':'str', 'genres':'str'})
rate_df = pd.read_csv("..\data\mov_ratings.csv", 
					header = 0,
					dtype={'userId':'int32', 'movieId':'int32', 'rating':'float64'})
rate_df['timestamp'] = pd.to_datetime(rate_df.timestamp) #I like time stamps.  

# Saving for later when i want to group on decades. 
# rate_df['year'] = pd.DatetimeIndex(small_rate_df['timestamp']).year
# X = rate_df.drop(["rating"], axis=1)
# y = rate_df['rating']


#Poking around the movie df
# mov_df.describe()
# mov_df.columns
# mov_df.isnull().sum()

#How many movies are there?
num_items = len(rate_df['movieId'].unique())
print("How many movies were there\n", num_items)

# rate_df.describe()
# rate_df.columns
# rate_df.isnull().sum()
#Yay no nulls

# How many user rated a movie?
num_users = len(rate_df['userId'].unique())
print("\nHow many users rated a movie\n", num_users)
# Count how many reviews per user
print("\nWhats their distribution of how many times they rated a movie look like?\n\n", rate_df.groupby('userId').size())

plt.figure(figsize=(16, 16))
plt.hist(rate_df['userId'], bins=100)
plt.ylim(ymin=200000)
plt.show()

#Sparcity
sparsity=round(1.0-len(rate_df)/float(num_users*num_items),3)
print("\n what kind of sparcity do we have?\n", str(sparsity*100) + '%' )


#%%

# Lets look at ratings for shits and giggles. 
mov_rating = rate_df.groupby(['movieId'], as_index=False)
avg_rating = mov_rating.agg({'rating':'mean'})
avg_rating.sort_values('rating', ascending=False)


#%%
#
# print("Time to calc SVD is:\n")
# @%%time

# TODO SCIPY SVD MODEL

#First, lets pivot this df so the user is rows and the movid id is the columns. 
# Looking at a smaller subset first as doing the whole thing breaks my comp

#Lets split up the data. 

small_rate_df = rate_df.iloc[:200000, :]  # Max i can do is 1000000 before it breaks
small_rate_pivot = small_rate_df.pivot(index='userId', 
										columns='movieId', 
										values='rating').fillna(0)


rating_matrix = small_rate_pivot.to_numpy()   # make it a matrix
rating_mean = np.mean(rating_matrix, axis=1)  # take the mean
rate_demeaned = rating_matrix - rating_mean.reshape(-1, 1) # subtract mean off matrix

#SVD magic here
U, sigma, Vt = svds(rate_demeaned, k=50)
#Documentation told me so  
sigma = np.diag(sigma)

predictions_mat = np.dot(np.dot(U, sigma), Vt) + rating_mean.reshape(-1, 1)

#Back to a dataframe. 
results_df = pd.DataFrame(predictions_mat, columns=small_rate_pivot.columns)

# %%

# TODO RMSE Calc
def rmse(orig, predictions):
	x = orig - predictions
	return sum([y*y for y in x])/len(x)
	return sqrt()
rmse_result = rmse(rating_matrix, predictions_mat)
print("The RMSE for the SVD model is {}".format(sum(rmse_result)/len(rmse_result)))


# Expected value for each user
# Go through original rating matrix to predictions
# %%

# %%
# when i calc a value of 5, what is it in expected predictions. 


# David idea
# Don't include someone who hasn't made a review in 25 years. 
# remove last 5 entries. 
# train 

# Make a mini tool for mini experiments.  Only do whats possible. 
# 




# %%
