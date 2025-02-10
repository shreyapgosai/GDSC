# GDSC
HERE I HAVE MADE A PROJECT OF A MOVIE RECOMMENDATION SYSTEM 
here there are two methode 
contant based and collabrative.....

LIBRERIES....
here there are numpy,sklearn,panda used in this code



DATASET.....
tmdb_5000_credits
tmdb_5000_movies


@CODE:
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

    #for convert text to list form


##MAIN THOUGHT OF THE CODE....
here cosine methode we used basically it fatches the most important words from overview of every movie and tries to fix the movies in a vector space according to those words then it will find distance of different movies and will come to know how much these movies are similar...and according to these the similar type of movies are decided



@CODE FOR THIS


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
cv.fit_transform(new['tags']).toarray()
vector = cv.fit_transform(new['tags']).toarray()
vector.shape
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)
similarity
new[new['title'] == 'The Lego Movie'].index[0]
def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


here we use sklearn to use its feature countvextorizer and we get the distances...




##HOW TO RUN 

use streamlit and create app
and then submit the link of git hub and it will scan the backend first and after this we can run frontend

    
