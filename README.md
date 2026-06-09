# 🎬 MOVIE RECOMMENDATION SYSTEM

---

## 📌 Table of Contents
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
- Team
- License

---

## 📖 Overview

The Movie Recommendation System is a machine learning-based content filtering system that recommends movies based on similarity of content features.

It uses movie metadata such as overview, genres, keywords, cast, and crew to compute similarity between movies and suggest relevant recommendations.

The system is designed to mimic real-world recommendation engines like Netflix.

---

## ✨ Features

- Smart Data Processing: Cleans and merges movie datasets  
- Multifeature Analysis: Uses genres, keywords, cast, crew, overview  
- Text-Based Representation: Converts text into numerical vectors  
- Similarity-Based Recommendations: Uses cosine similarity  
- Fast Retrieval: Efficient ranking of similar movies  
- Simple Web Interface: Built using Streamlit  

---

## 🏗️ System Architecture

```text
Input Movie Title
        ↓
Feature Extraction (Genres, Keywords, Cast, Crew, Overview)
        ↓
Data Cleaning & Processing
        ↓
Tag Generation
        ↓
Text Vectorization (CountVectorizer)
        ↓
Cosine Similarity Matrix
        ↓
Recommendation Engine
        ↓
Top 5 Similar Movies Output
