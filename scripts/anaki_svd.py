#%%
# Yes i used pandas.  Don't hate me because i'm beautiful
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse.linalg import svds
import timeit

# mov_df = pd.read_csv("..\data\movies.csv", 
# 					header = 0,
# 					dtype={'movieId':'int32', 'title':'str', 'genres':'str'})


# TODO read function

rate_df = pd.read_csv("..\data\mov_ratings.csv", 
					header = 0,
					dtype={'userId':'int32', 'movieId':'int32', 'rating':'float64'})
rate_df['timestamp'] = pd.to_datetime(rate_df.timestamp) #I like time stamps.  


#%%
# Creating 70/30 test train split
train_data, test_data = train_test_split(rate_df, test_size = 0.3, random_state = 42)

num_items = len(rate_df['movieId'].unique())
num_users = len(rate_df['userId'].unique())


# Making two matrices out of training and test data. 
train_matrix = train_data.iloc[:100000, :]
train_matrix_pivot = train_matrix.pivot(index='userId', 
										columns='movieId', 
										values='rating').fillna(0)


# test_matrix = test_data.iloc[:200000, :]
# test_matrix = test_matrix.pivot(index='userId', 
# 										columns='movieId', 
# 										values='rating').fillna(0)


#%%
# TODO SCIPY SVD MODEL

# TODO Need a user based recoomendation AND item based reccomender
 

# TODO SCIPY SVD MODEL


# small_rate_df = rate_df.iloc[:1000000, :]
# small_rate_pivot = small_rate_df.pivot(index='userId', 
# 										columns='movieId', 
# 										values='rating').fillna(0)


rating_matrix = train_matrix_pivot.to_numpy()    #make it a matrix
rating_mean = np.mean(rating_matrix, axis=1)  #Take the mean
rate_demeaned = rating_matrix - rating_mean.reshape(-1, 1) # subtract mean off matrix

#SVD magic here
U, sigma, Vt = svds(rate_demeaned, k=50)
#Documentation told me so
sigma = np.diag(sigma)

predictions_mat = np.dot(np.dot(U, sigma), Vt) + rating_mean.reshape(-1, 1)

#Back to a dataframe. 
results_df = pd.DataFrame(predictions_mat, columns=train_matrix_pivot.columns)

# %%
# TODO RMSE Calc
def rmse(orig, predictions):
	x = orig - predictions
	return sum([y*y for y in x])/len(x)

rmse_result = rmse(rating_matrix, predictions_mat)
print("The RMSE for the SVD model is {}".format(sum(rmse_result)/len(rmse_result)))

# %%
