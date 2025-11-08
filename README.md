Amazon Sentiment Analysis System

This project compares customer sentiments from two Amazon datasets Fashion Reviews and Luxury Beauty Reviews using NLP techniques. It classifies reviews into Positive, Neutral, and Negative sentiments and visualizes patterns through an interactive Streamlit web app. The system helps understand how customer opinions differ across product categories and highlights key insights using word clouds and charts.

Features

Upload and analyze two datasets simultaneously (Fashion & Luxury)

Automatic data cleaning and preprocessing

VADER-based sentiment classification

Comparison of sentiment distribution across categories

Word Clouds for Positive and Negative sentiments

Interactive visualizations using Plotly

Downloadable processed CSV files

Clean and modern UI with category-specific insights

Tech Stack

Python

Pandas

NLTK / VADER

Matplotlib & WordCloud

Plotly

Streamlit

Project Workflow

Data Loading
Reads two review datasets and extracts review text columns.

Data Cleaning & Preprocessing
Handling missing values, converting text to lowercase, and preparing text for analysis.

Sentiment Analysis
Uses VADER to classify reviews into Positive, Neutral, or Negative.

Visualization & Insights
Interactive charts, word clouds, and category-wise comparisons.

Download Output
Export processed sentiment data for further analysis.

Screenshots 



Conclusion

This project provides a complete end-to-end pipeline for sentiment analysis on Amazon reviews, offering comparisons between Fashion and Luxury product categories. It shows how NLP, data visualization, and interactive dashboards can convert raw text into insights useful for product evaluation and customer experience analysis.
