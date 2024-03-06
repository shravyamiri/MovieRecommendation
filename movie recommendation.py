#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import sklearn
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[3]:


Rating = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/ratings.csv")
print(Rating.head())


# In[4]:


Movie = pd.read_csv("https://s3-us-west-2.amazonaws.com/recommender-tutorial/movies.csv")
print(Movie.head())


# In[5]:


Rating_s = len(Rating)
Movie_s= len(Rating['movieId'].unique())
User_s = len(Rating['userId'].unique())
print(f"No.of ratings: {Rating_s}")
print(f"No.of distinct Movie id's: {Movie_s}")
print(f"No.of distinct users: {User_s}")
print(f"Avg ratings per user: {round(Rating_s/User_s, 2)}")
print(f"Avg  ratings per movie: {round(Rating_s/Movie_s, 2)}")


# In[6]:


user_frequency = Rating[['userId', 'movieId']].groupby(
	'userId').count().reset_index()
user_frequency.columns = ['userId', 'Rating_s']
print(user_frequency.head())


# In[10]:


mean_Rating = Rating.groupby('movieId')[['Rating']].mean()
lowest_rated = mean_Rating['rating'].idxmin()
Movie.loc[Movie['movieId'] == lowest_rated]
highest_rated = mean_Rating['rating'].idxmax()
Movie.loc[Movie['movieId'] == highest_rated]
Rating[Rating['movieId']==highest_rated]
Rating[Rating['movieId']==lowest_rated]
Movie_stats = Rating.groupby('movieId')[['rating']].agg(['count', 'mean'])
Movie_stats.columns = Movie_stats.columns.droplevel()


# In[11]:


from scipy.sparse import csr_matrix
def create_matrix(df):
	A= len(df['userId'].unique())
	B = len(df['movieId'].unique())
	user_mapper = dict(zip(np.unique(df["userId"]), list(range(A))))
	movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(B))))
	user_inv_mapper = dict(zip(list(range(A)), np.unique(df["userId"])))
	movie_inv_mapper = dict(zip(list(range(B)), np.unique(df["movieId"])))
	user_index = [user_mapper[i] for i in df['userId']]
	movie_index = [movie_mapper[i] for i in df['movieId']]
	C = csr_matrix((df["Rating"], (movie_index, user_index)), shape=(B, A))
	return C, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper
C, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(Rating)


# In[15]:


from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import NearestNeighbors
def find_similar_movies(movie_id, X, k, metric='cosine', show_distance=False):
	neighbour_ids = []
	movie_ind = movie_mapper[movie_id]
	movie_vec = X[movie_ind]
	k+=1
	knn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
	knn.fit(X)
	movie_vec = movie_vec.reshape(1,-1)
	neighbour = knn.kneighbors(movie_vec, return_distance=show_distance)
	for i in range(0,k):
		n = neighbour.item(i)
		neighbour_ids.append(movie_inv_mapper[n])
	neighbour_ids.pop(0)
	return neighbour_ids
movie_titles = dict(zip(Movie['movieId'], Movie['title']))
movie_id = 3
similar_ids = find_similar_movies(movie_id, C, k=10)
movie_title = movie_titles[movie_id]
print(f"Since you watched {movie_title}")
for i in similar_ids:
	print(movie_titles[i])


# In[10]:


def recommend_movies_for_user(user_id, X, user_mapper, movie_mapper, movie_inv_mapper, k=10):
	df1 = ratings[ratings['userId'] == user_id]
	
	if df1.empty:
		print(f"User with ID {user_id} does not exist.")
		return

	movie_id = df1[df1['rating'] == max(df1['rating'])]['movieId'].iloc[0]

	movie_titles = dict(zip(movies['movieId'], movies['title']))

	similar_ids = find_similar_movies(movie_id, X, k)
	movie_title = movie_titles.get(movie_id, "Movie not found")

	if movie_title == "Movie not found":
		print(f"Movie with ID {movie_id} not found.")
		return

	print(f"Since you watched {movie_title}, you might also like:")
	for i in similar_ids:
		print(movie_titles.get(i, "Movie not found"))


# In[11]:


user_id = 150 # Replace with the desired user ID
recommend_movies_for_user(user_id, X, user_mapper, movie_mapper, movie_inv_mapper, k=10)


# In[12]:


user_id = 2300 # Replace with the desired user ID
recommend_movies_for_user(user_id, X, user_mapper, movie_mapper, movie_inv_mapper, k=10)


# In[ ]:




