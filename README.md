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

## TEXT VECTORIZATION (COUNT VECTORIZER)

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')

vector = cv.fit_transform(new['tags']).toarray()
