#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go


mov_df = pd.read_csv("..\data\movies.csv", 
					header = 0,
					dtype={'movieId':'int32', 'title':'str', 'genres':'str'})
rate_df = pd.read_csv("..\data\mov_ratings.csv", 
					header = 0,
					dtype={'userId':'int32', 'movieId':'int32', 'rating':'float64'})
rate_df['timestamp'] = pd.to_datetime(rate_df.timestamp) #I like time stamps.  


#%%
#Movie Genre plotly
# Count the number of genres in each 'genre' and sort them in descending order

#color pallette

num_genres_in_category = mov_df['genres'].value_counts().sort_values(ascending = False)
num_genres_in_category = num_genres_in_category[num_genres_in_category>=100] #subset only categories with more than 100 labels


fig = go.Figure(
    data=[go.Bar(x=num_genres_in_category.index, y=num_genres_in_category.values)],
    layout=go.Layout(
        title=go.layout.Title(text="Count of Movie Genres")
    )
)

plotly.offline.iplot(fig, filename='genres.html')

#%%
# Average rating of apps
# rate_df = rate_df.iloc[:25000000,:] #20000000
avg_mov_rating = rate_df["rating"].mean()
print('Average movie rating = ', avg_mov_rating)

# Distribution of movie rating according to their ratings
data = [go.Histogram(
		x = rate_df['rating'],
		xbins=dict(start='0', end='5', size=0.4)
)]

layout = go.Layout(
		title = 'Histogram of Ratings',
		xaxis = go.XAxis(title='Ratings'),
		yaxis = go.YAxis(title='Count'), 
		shapes= [{  # Vertical dashed line to indicate the average movie rating
			  'type' :'line',
			  'x0': avg_mov_rating,
			  'y0': 0,
			  'x1': avg_mov_rating,
			  'y1': 6000,
			  'line': { 'dash': 'dashdot'}}],
		annotations=[
            go.layout.Annotation(
                text='Average Movie Rating of:{}'.format(avg_mov_rating),
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.8,
                y=1.1,
                bordercolor='blue',
                borderwidth=1
            )]
)


plotly.offline.iplot({'data': data, 'layout': layout}, filename='ratings.html')

# %%

genre_rating_df = pd.read_csv("..\data\genre_rating.csv", 
					header = 0,
					sep=",",
					index_col=['ratingYear'])

fig = go.Figure(
	    layout=go.Layout(
        	title=go.layout.Title(text="Count of Movie Genres"),
			xaxis=go.layout.XAxis(title="Year"),
			yaxis=go.layout.YAxis(title="Rating")
    )
)

for col in genre_rating_df.columns:
    fig.add_trace(go.Scatter(x=genre_rating_df.index, 
							 y=genre_rating_df[col].values,
                             name = col,
                             mode = 'markers+lines',
                             line=dict(shape='linear'),
                             connectgaps=True
                             )
                 )
plotly.offline.iplot(fig, filename='genre_ratings.html')
# %%
