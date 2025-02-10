import numpy as np 
import pandas as pd 
import sklearn 
movies = pd.read_csv('https://raw.githubusercontent.com/shreyapgosai/GDSC/main/tmdb_5000_movies.csv')
credits = pd.read_csv('https://raw.githubusercontent.com/shreyapgosai/GDSC/main/tmdb_5000_credits.zip') 


movies = movies.merge(credits,on='title')
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

import ast
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 
movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)

movies['keywords'] = movies['keywords'].apply(convert)

import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 
movies['cast'] = movies['cast'].apply(convert)
movies.head()
movies['cast'] = movies['cast'].apply(lambda x:x[0:3])
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 
movies['crew'] = movies['crew'].apply(fetch_director)
#movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)
movies['genres']=movies['genres'].apply(lambda x:[i.replace("","") for i in x])
movies['keywords']=  movies['keywords'].apply(lambda x:[i.replace("","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace("","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace("","") for i in x])
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies.head()
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
#new.head()
new['tags'] = new['tags'].apply(lambda x: " ".join(x))


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

if 'tags' in new.columns:
    new['tags'] = new['tags'].fillna('')  # Replace NaN values with empty strings
    vector = cv.fit_transform(new['tags']).toarray()
else:
    raise ValueError("Column 'tags' not found in the dataframe 'new'")

def stem(text):
    y=[]
    for i in text.split:
        ps.stem(i)
        vector = cv.fit_transform(new['tags']).toarray()
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)
similarity
new[new['title'] == 'The Lego Movie'].index[0]
def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)
        new['title'].values

recommend('Avatar')
import streamlit as st
import pickle
import pandas as pd
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        movie_id=i[0]


        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies

# movies_dict=pickle.load(open('movie_dict.pkl','rb'))
# movies=pd.DataFrame(movies_dict)
# similarity=pickle.load(open('similarity.pkl','rb'))

st.title('movie recommendation System')
option = st.selectbox(
    "How would you like to be contacted?",
    movies['title'].values
)
if st.button('recommend'):
    recommendations=recommend(option)
    for i in recommendations:
        st.write(i)
