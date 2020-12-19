#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

mov_df = pd.read_csv("..\data\movies.csv", header = 0)
rate_df = pd.read_csv("..\data\mov_ratings.csv", header = 0)

#Poking around the movie df
mov_df.describe()
mov_df.columns
mov_df.isnull().sum()


rate_df.describe()
rate_df.columns
rate_df.isnull().sum()
#Yay no nulls


