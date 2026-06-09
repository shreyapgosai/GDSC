# MOVIE RECOMMENDATION SYSTEM

---

## TABLE OF CONTENTS

- Overview
- Features
- System Architecture
- Key Components
- Data Processing & Feature Engineering
- Text Vectorization (CountVectorizer)
- Similarity Computation (Cosine Similarity)
- Recommendation Engine
- UI/UX and Web Deployment
- Installation and Usage
- Prerequisites
- Setup
- Run Application
- Team
- License

---

## OVERVIEW

The Movie Recommendation System is a machine learning-based content filtering system that recommends movies based on similarity of content features.

It uses movie metadata such as overview, genres, keywords, cast, and crew to compute similarity between movies and suggest relevant recommendations.

The system is designed to mimic real-world recommendation engines like Netflix.

---

## FEATURES

- Smart data preprocessing and cleaning  
- Uses multiple features like genres, keywords, cast, crew, overview  
- Converts text data into numerical vectors  
- Uses cosine similarity for recommendation  
- Fast and efficient retrieval of similar movies  
- Streamlit-based interactive web application  

---

## SYSTEM ARCHITECTURE

Input Movie Title  
→ Feature Extraction (Genres, Keywords, Cast, Crew, Overview)  
→ Data Cleaning and Processing  
→ Tag Generation  
→ Text Vectorization (CountVectorizer)  
→ Cosine Similarity Matrix  
→ Recommendation Engine  
→ Top 5 Similar Movies Output  

---

## KEY COMPONENTS

---

## DATA PROCESSING & FEATURE ENGINEERING

Movies dataset is cleaned and transformed by extracting important features:

- Genres  
- Keywords  
- Cast (Top 3 actors)  
- Crew (Director)  
- Overview  

All features are combined into a single column called tags.

---
## SIMILARITY COMPUTATION (COSINE SIMILARITY)

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vector)

This computes similarity between all movies using cosine similarity on the vectorized tags. It measures how similar two movies are based on their feature vectors in multidimensional space.

---

## RECOMMENDATION ENGINE



    

This function:
- Finds the index of the selected movie
- Computes similarity scores with all other movies
- Sorts movies based on similarity score
- Returns top 5 most similar movies

---

## UI/UX AND WEB DEPLOYMENT

- Built using Streamlit  
- Uses dropdown menu for movie selection  
- Button-based recommendation system  
- Displays top recommended movies  
- Simple and interactive user interface  

---

## INSTALLATION AND USAGE

### PREREQUISITES

- Python 3.8 or higher  
- numpy  
- pandas  
- sklearn  
- streamlit  

---

### SETUP

git clone https://github.com/your-username/movie-recommendation-system.git  
cd movie-recommendation-system  
pip install -r requirements.txt  

---

## RUN APPLICATION

streamlit run app.py  

Open in browser:

http://localhost:8501  

---

## TEAM

Shreya Gosai
