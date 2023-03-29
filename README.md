# Data-Science

The project is a sentiment analysis project that involves collecting tweets using the Twitter API and analyzing them to determine their sentiment. The goal of the project is to classify the tweets into positive, negative, and neutral categories based on the content of the tweets.


The project involves several steps, including data collection using Tweepy, data preprocessing using the NLTK toolkit, and model building using machine learning algorithms such as SVM, Naive Bayes, and logistic regression. The data is cleaned by removing stop words, punctuation, and emojis, and lowering the case of the words.
The sentiment analysis is performed using the sentiment score of the tweets, where a positive score corresponds to a positive sentiment, a negative score corresponds to a negative sentiment, and a neutral score corresponds to a neutral sentiment. The sentiment score is added as a column to the dataset.


The project is a multiclass classification problem as it involves classifying tweets into three categories. SVM and Naive Bayes are used for model building, as they are effective algorithms for multiclass classification problems.


Overall, the project aims to provide insights into the sentiment of tweets related to a particular topic, which can be useful for businesses, organizations, and individuals in making decisions and understanding public opinion.

Steps involved to build a machine learning model:

1. Data Collection
2. Preparing the data
3. Data Preprocessing
4. Training the model
5. Evaluating the model
6. Making Predictions

The various steps involved in our project workflow are:
Import Necessary Dependencies
Read and Load the Dataset
Exploratory Data Analysis
Data Visualization of Target Variables
Data Preprocessing
Splitting our data into Train and Test sets.
Transforming Dataset using TF-IDF Vectorizer
Function for Model Evaluation
Model Building
Model Evaluation


#DATA COLLECTION:

For data collection in this project, I used the Tweepy library to retrieve live tweets from Twitter based on specific keywords and the desired number of tweets. To access the Twitter API, I created a Twitter developer account and obtained the necessary access keys and tokens. After that, I created a Python script using Tweepy to connect to the Twitter API and retrieve tweets containing the target keywords. Overall, the Tweepy library and the Twitter API made the data collection process efficient and straightforward, allowing me to obtain a sufficient amount of high-quality data for analysis.


Importing necessary dependencies:


Tweepy: Tweepy is an open-source Python package that provides easy access to the Twitter API. It allows you to authenticate, search for tweets, and retrieve data from Twitter.


Pandas: Pandas is a powerful Python library for data manipulation and analysis. It is used to handle and organize data in a tabular format, which is useful for storing and analyzing large datasets.
NumPy: NumPy is a popular Python library for scientific computing. It is used for mathematical operations on arrays and matrices, making it an essential tool for data analysis.


Matplotlib: Matplotlib is a plotting library for Python. It is used to create a variety of visualizations, including line plots, scatter plots, and histograms, which are useful for exploring and visualizing data.


NLTK: NLTK stands for Natural Language Toolkit. It is a Python library for working with human language data, including text processing, tokenization, stemming, and more. It is used for preprocessing the raw text data collected from Twitter.
Scikit-learn: Scikit-learn is a popular Python library for machine learning. It provides a variety of tools for data preprocessing, model selection, and evaluation, making it a valuable tool for building and evaluating machine learning models.


We are going to extract the live tweets to prepare our own dataset. Using the nltk toolkit we assign a sentiment score to respective tweets. This helps us to know the sentiment of an individual tweet.
This is just a sample hashtag I used to retrieve the tweets. I took in 2000 tweets, this depends on our processing power, and tweepy has a restriction to retrieve a certain number of tweets in one take. Or else the account will be suspended.

