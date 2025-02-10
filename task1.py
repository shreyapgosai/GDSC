import numpy as np 
import pandas as pd 
import sklearn 
movies = pd.read_csv('https://raw.githubusercontent.com/shreyapgosai/GDSC/main/tmdb_5000_movies.csv')
credits = pd.read_csv('https://raw.githubusercontent.com/shreyapgosai/GDSC/main/tmdb_5000_credits.zip') 
movies.head(2)
movies.shape
credits.head()
movies = movies.merge(credits,on='title')
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.head()
import ast
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 
movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies.head()
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()
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
new.head()
new['tags'][0]
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
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
spark = SparkSession.builder.appName("ALS_Recommender").getOrCreate()
data = [
    (0, 10, 4.0), (0, 20, 3.5), (0, 30, 5.0),
    (1, 10, 2.0), (1, 20, 4.5), (1, 30, 3.0),
    (2, 10, 5.0), (2, 20, 3.5), (2, 30, 4.0),
]
columns = ["userId", "movieId", "rating"]
ratings_df = spark.createDataFrame(data, columns)

# Train ALS model
als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    nonnegative=True,
    implicitPrefs=False,
    rank=10,
    maxIter=10,
    regParam=0.1
)
model = als.fit(ratings_df)


def get_cf_score(user_id, movie_id):
    """ Predict rating for a given user and movie """
    user_df = spark.createDataFrame([(user_id, movie_id)], ["userId", "movieId"])
    prediction = model.transform(user_df).collect()
    return prediction[0]["prediction"] if prediction else 0 
def recommend(movie):
    """ Content-Based Filtering function (same as your original code) """
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    return [movies.iloc[i[0]].title for i in distances[1:6]]

def hybrid_recommend(user_id, movie):
    """ Combines Content-Based and ALS Collaborative Filtering """
    content_rec = recommend(movie)  # Get content-based recommendations
    movie_ids = movies[movies['title'].isin(content_rec)]['movie_id'].tolist()

    # Get CF Scores using ALS
    cf_scores = [get_cf_score(user_id, mid) for mid in movie_ids]

    # Compute Hybrid Scores (50% CF + 50% Content Similarity)
    hybrid_scores = np.array(cf_scores) * 0.5 + np.array([
        similarity[movies[movies['title'] == movie].index[0]][movies[movies['movie_id'] == mid].index[0]]
        for mid in movie_ids
    ]) * 0.5

    ranked_movies = [x for _, x in sorted(zip(hybrid_scores, content_rec), reverse=True)]
    return ranked_movies[:5]

recommend('Avatar')
print(hybrid_recommend(user_id=6, movie='Avatar'))
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

movies_dict=pickle.load(open('movie_dict.pkl','rb'))
movies=pd.DataFrame(movies_dict)
similarity=pickle.load(open('similarity.pkl','rb'))

st.title('movie recommendation System')
option = st.selectbox(
    "How would you like to be contacted?",
    movies['title'].values
)
if st.button('recommend'):
    recommendations=recommend(option)
    for i in recommendations:
        st.write(i)
