"""
Created on Sat Dec 19, 2020

@author: Christian Nava
"""

import numpy as np
import pandas as pd

# read in genome score and tag data
genome_scores = pd.read_csv('../ml-25m/genome-scores.csv')
genome_tags = pd.read_csv('../ml-25m/genome-tags.csv')

# join datasets
genome_tags_and_scores = pd.merge(genome_scores, genome_tags, on='tagId', how='left')

# rename tag column for clarity
genome_tags_and_scores = genome_tags_and_scores.rename(columns = {"relevance":"tag_relevance"})

# get the 5 most relevant tags for each movieId
# value in .tail() controls number of relevant tags returned
relevant_tags = genome_tags_and_scores.sort_values(['movieId', 'tag_relevance']).groupby('movieId').tail(5)

#relevant_tags.head(15)

#pd.to_pickle(relevant_tags,'./relevant_tags.pkl') 


### Create Plot ###

import matplotlib.pyplot as plt
import seaborn as sns

# Set figure size
plt.figure(figsize = (15, 10))

relevant_tags['tag'].value_counts()[:20].sort_values().plot(kind='barh')
plt.title('Top 20 Most Relevant Tags', fontsize = 20) 
plt.xlabel('Count of Movies Where That Tag is One of the 5 Most Relevant', fontsize = 16) 
plt.ylabel('User Generated Tag', fontsize = 16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=12)