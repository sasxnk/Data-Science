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


DATA COLLECTION:

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


![image](https://user-images.githubusercontent.com/83261877/228514220-3b0a0905-ab2d-4834-a1e7-f5e19f9ff4c1.png)

![image](https://user-images.githubusercontent.com/83261877/228513904-70400356-707f-472d-8598-ff7d07edf358.png)

![image](https://user-images.githubusercontent.com/83261877/228514329-bb215def-e4a2-459f-922a-41255c18b506.png)

![image](https://user-images.githubusercontent.com/83261877/228514788-98e56a9a-24dd-4a2f-b2ec-3d7640dbe5ab.png)

![image](https://user-images.githubusercontent.com/83261877/228514872-229b25cb-9e6e-48e4-a8b5-5b5c717da1ef.png)

![image](https://user-images.githubusercontent.com/83261877/228515038-df9d9310-263d-4c1c-a147-78c4b09d4339.png)

![image](https://user-images.githubusercontent.com/83261877/228515213-12b0cb62-5153-4719-9aeb-a51384411539.png)

![image](https://user-images.githubusercontent.com/83261877/228515387-86034ba9-17f9-4b41-b30f-1e29e10915cd.png)



we are going to set up the Twitter API authentication using the provided consumer key, consumer secret, access token, and access token secret. It then prompts the user to enter a keyword or hashtag to search and the number of tweets to analyze. It uses the Tweepy library to search for the specified number of tweets containing the keyword or hashtag and stores them in a pandas DataFrame called tweet_list.


The sentiment analysis is performed on each tweet using TextBlob and SentimentIntensityAnalyzer from the NLTK library. The sentiment score for each tweet is assigned as either positive (4), negative (0), or neutral (2) based on the compound score returned by SentimentIntensityAnalyzer. The percentage of positive, negative, and neutral tweets and the overall polarity score are calculated and printed.


Finally, the number of positive, negative, and neutral tweets is printed along with the total number of tweets analyzed.


The code segment first performs some basic operations on the tweet_list dataframe, including checking its shape, info, and data types. Then, it converts the sentiment_score column to an integer data type and checks the unique values of that column. It then creates a bar plot of the sentiment score distribution using the Seaborn library. The code then creates three new dataframes based on the sentiment scores (positive, negative, and neutral). It concatenates these dataframes to create a new dataset. Finally, it converts all the text in the 'text' column to lowercase.


This is a code that involves several text preprocessing steps to clean up a dataset's textual data. The code begins by defining a list of stop words that will be removed from the text data. Then, a function is defined to remove the stop words from the text data. The next step involves removing punctuation from the text data using a similar function. The following steps remove repeating characters, URLs, and numbers from the text data. Then, the data is tokenized using the regular expression tokenizer. Next, stemming is performed using the Porter stemming algorithm, followed by lemmatization using the WordNet lemmatizer. Each step modifies the dataset's 'text' column using the 'apply' function.


This code seems to be training and evaluating two different models, Bernoulli Naive Bayes and Linear Support Vector Classifier (SVC), for sentiment analysis task. The code first separates the input text data and corresponding sentiment scores into training and testing sets using train_test_split method from sklearn.model_selection. Then it applies TF-IDF vectorization on the text data using TfidfVectorizer from sklearn.feature_extraction.text. The resulting sparse matrix is used to train the two models, and evaluate their performance using classification_report and confusion_matrix from sklearn.metrics. The confusion matrix is also plotted as a heatmap using seaborn library.


The Bernoulli Naive Bayes model is trained using BernoulliNB from sklearn.naive_bayes and the Linear SVC model is trained using LinearSVC from sklearn.svm. The trained models are evaluated on the test data by calling the model_Evaluate function which prints out the classification report and plots the confusion matrix. Finally, the predictions made by both models on the test data are stored in y_pred1 and y_pred2 respectively.

![image](https://user-images.githubusercontent.com/83261877/228515600-f1a43b5f-00d3-49fb-87e3-650228c07975.png)

![image](https://user-images.githubusercontent.com/83261877/228515700-6d8ce450-7e15-4851-8f3f-5160190588fb.png)

