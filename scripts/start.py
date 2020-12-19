import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import  SVC

movies = pd.read_csv('..\data\movies.csv', header = 0)
ratings = pd.read_csv('..\data\ratings.csv', header = 0)

