MOVIE RECOMMENDATION SYSTEM

Table of Contents
Overview
Features
System Architecture
Key Components

Data Processing & Feature Engineering
Text Vectorization (CountVectorizer)
Similarity Computation (Cosine Similarity)
Recommendation Engine
UI/UX and Web Deployment
Installation and Usage
Prerequisites
Setup
Team
License
Overview

The Movie Recommendation System is a machine learning-based content filtering system that recommends movies based on similarity of content features.

It uses movie metadata such as overview, genres, keywords, cast, and crew to compute similarity between movies and suggest relevant recommendations.

The system is designed to mimic real-world recommendation engines like Netflix.

Features

Smart Data Processing: Cleans and merges movie datasets
Multifeature Analysis: Uses genres, keywords, cast, crew, overview
Text-Based Representation: Converts text into numerical vectors
Similarity-Based Recommendations: Uses cosine similarity
Fast Retrieval: Efficient ranking of similar movies
Simple Web Interface: Built using Streamlit

System Architecture

Input Movie Title
→ Feature Extraction (Genres, Keywords, Cast, Crew, Overview)
→ Data Cleaning & Processing
→ Tag Generation
→ Text Vectorization (CountVectorizer)
→ Cosine Similarity Matrix
→ Recommendation Engine
→ Top 5 Similar Movies Output

Key Components
1. Data Processing & Feature Engineering

Movies dataset is cleaned and transformed by extracting useful features:

Genres
Keywords
Cast (Top 3 actors)
Crew (Director)
Overview

All features are combined into a single text field called tags.

2. Text Vectorization (CountVectorizer)

Text data is converted into numerical vectors using:

CountVectorizer(max_features=5000, stop_words='english')

This converts movie tags into a matrix of word frequency representation.

3. Similarity Computation (Cosine Similarity)

Similarity between movies is calculated using cosine similarity:

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)

This measures how close two movies are in vector space.

4. Recommendation Engine

def recommend(movie):
index = new[new['title'] == movie].index[0]
distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
for i in distances[1:6]:
print(new.iloc[i[0]].title)

The system returns the top 5 most similar movies.

UI/UX and Web Deployment
Built using Streamlit
Dropdown menu for movie selection
One-click recommendation button
Displays top recommended movies
Simple and interactive UI
Installation and Usage
Prerequisites

Python 3.8+
Libraries: numpy, pandas, sklearn, streamlit

Setup

git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system

pip install -r requirements.txt

Run Application

streamlit run app.py

Then open:
http://localhost:8501

Team

Shreya Gosai
