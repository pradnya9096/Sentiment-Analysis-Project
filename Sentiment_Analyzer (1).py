#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 10:13:46 2023

@author: tarunvannelli
"""
import re
from nltk.corpus import wordnet
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from afinn import Afinn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import streamlit as st
import base64
import pandas as pd
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')


df = pd.read_csv("/Users/tarunvannelli/Amazon_Product_Reviews.csv")
df = df.dropna(subset=['title', 'body'])
df['text'] = df['title'] + ' ' + df['body']


# find sentences containing HTML tags

i = 0
for sent in df['text'].values:
    if (len(re.findall('<.*?>', sent))):
        print(i)
        print(sent)
        break
    i += 1

data = df['text']

# Converting text to lower
data = df['text'].apply(lambda x: x.lower())

# Remove special characters and punctuation


def remove_special_chars(text):
    # Remove special characters
    text = re.sub('[^a-zA-Z0-9\s]', '', text)

    # Remove punctuation
    text = re.sub('[^\w\s]', '', text)

    return text


data = data.apply(remove_special_chars)

# assuming 'data' is a Series object
df = data.to_frame()

for index, row in df.iterrows():
    # Here we are filtering out all the words that contains link
    words_without_links = [
        word for word in row.text.split() if 'http' not in word]
    df.at[index, 'text'] = ' '.join(words_without_links)

data = pd.Series(df['text'].values, index=df.index)

# Function for POS


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# Tokenization


nltk.download('punkt')

# download stopwords
nltk.download('stopwords')

# import stopwords and word_tokenize from NLTK

# create a list of stopwords
stop_words = set(stopwords.words('english'))

# tokenize the text column into individual words
data = data.apply(lambda x: word_tokenize(x))

# remove stopwords
tokenized_data = data.apply(
    lambda x: [word for word in x if word not in stop_words])

# Lemmatiztion

nltk.download('omw-1.4')
nltk.download('wordnet')


# create a lemmatizer object
lemmatizer = WordNetLemmatizer()

# apply lemmatizing to each word in the data
lematized_data = tokenized_data.apply(
    lambda x: [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in x])

# Spearating positive and negative Words


# initialize Afinn
af = Afinn()

# initialize lists to store sentiment scores and sentiment categories
sentiment_scores = []
sentiment_categories = []

# iterate over each review and calculate its sentiment score
for review in lematized_data:
    if isinstance(review, list):  # check if review is a list
        review = ' '.join(review)  # join list of words into a single string
    sentiment_score = af.score(review)
    sentiment_scores.append(sentiment_score)

    # classify each review as positive, negative, or neutral based on its sentiment score
    if sentiment_score > 0:
        sentiment_categories.append('positive')
    elif sentiment_score < 0:
        sentiment_categories.append('negative')
    else:
        sentiment_categories.append('neutral')

# count number of reviews in each sentiment category
positive_count = sentiment_categories.count('positive')
negative_count = sentiment_categories.count('negative')
neutral_count = sentiment_categories.count('neutral')

# print results
print('Positive reviews:', positive_count)
print('Negative reviews:', negative_count)
print('Neutral reviews:', neutral_count)

# TF-TDF

# join the lists of lemmatized words for each document into a single string
lm_data = lematized_data.apply(lambda x: ' '.join(x))

# create a TfidfVectorizer object
vectorizer = TfidfVectorizer(max_features=1500, min_df=8, stop_words='english')

# fit the vectorizer to the preprocessed data
vectorizer.fit(lm_data)

# create TF-IDF vectors for the preprocessed data
tfidf_vectors = vectorizer.transform(lm_data)

# print the shape of the TF-IDF matrix
print(tfidf_vectors.shape)

# SPlitting Data


# create a new DataFrame with the preprocessed data and sentiment labels
# create X and y DataFrames
X = pd.DataFrame(tfidf_vectors.toarray(),
                 columns=vectorizer.get_feature_names_out())
y = pd.DataFrame(sentiment_categories, columns=['sentiment'])

# encode the sentiment labels as integers
le = LabelEncoder()
y = le.fit_transform(sentiment_categories)


# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# print the shapes of the train and test sets
print("Train set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Balancing Reviews
# Smote


# Resample training data using SMOTE
smote = SMOTE()
x_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Resample test data using SMOTE
x_test_resampled, y_test_resampled = smote.fit_resample(X_test, y_test)

# Convert resampled target vector to pandas Series
y_train_resampled_series = pd.Series(y_train_resampled)

# Check distribution of classes in resampled data
print(y_train_resampled_series.value_counts())

# Convert resampled target vector to pandas Series
y_test_resampled_series = pd.Series(y_test_resampled)

# Check distribution of classes in resampled data
print(y_test_resampled_series.value_counts())


# Model Building


# Create logistic regression model with best hyperparameters


lr_model = LogisticRegression(
    C=10, multi_class='ovr', penalty='l1', solver='liblinear')
# Fit model to data
lr = lr_model.fit(x_train_resampled, y_train_resampled)
y_pred_lr = lr.predict(x_test_resampled)
print(classification_report(y_test_resampled, y_pred_lr))


st.title(":red[Sentiment Analysis App] :smile: :neutral_face: :disappointed:")

st.write("Text to Analyze")

txt = st.text_area(
    'Enter Text', '''Hi I am really excited to present how good this sentiment analyzer works''')


def emoji_pattern(senti):
    if senti == 'positive':
        custom_emoji = "😄"
    elif senti == 'negative':
        custom_emoji = '😞'
    elif senti == 'neutral':
        custom_emoji = '😐'

    return custom_emoji


def run_sentiment_analysis(txt):

    sentences = nltk.sent_tokenize(txt)
    corpus = []
    for i in range(len(sentences)):
        review = re.sub('[^a-zA-Z]', ' ', sentences[i])
        review = review.lower()
        review = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in nltk.word_tokenize(
            review) if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)

    X = vectorizer.transform(corpus)
    predicted_sent = lr.predict(X)
    transform_sentiment = le.inverse_transform(predicted_sent)

    df_sent = pd.DataFrame()
    df_sent['text'] = sentences
    df_sent['sentiment'] = transform_sentiment
    df_sent['emoji_sentiment'] = df_sent['sentiment'].apply(emoji_pattern)

    agree = st.checkbox('View Sentiment for all Sentences')

    if agree:
        st.write(df_sent)

    if df_sent['sentiment'].value_counts().idxmax() == 'positive':
        st.markdown("![POSITIVE](https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExNWVlM2UzNjA2NmJjNmNmOTg2ZGE0ODNjZGZiMWQ2YTI0NzRhNTlhYyZjdD1n/xTiN0E03sgnvms9Uli/giphy.gif)")
    elif df_sent['sentiment'].value_counts().idxmax() == 'negative':
        st.markdown("![NEGATIVE](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExZGExZjU2MzQ2ZmM0N2MwZmY0MmRmYjhkNmI5Zjg2NzI4MDgyNTg4YSZjdD1n/StAnQV9TUCuys/giphy.gif)")
    else:
        st.markdown("![NEUTRAL](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExN2RkODJhNWQ2NGQ0NzI0ZjQ1MTNjNGE3MjhjYmU5MzUzYzc1ODY3YyZjdD1n/PQMJiBHKkGgbQiORyx/giphy.gif)")

    return df_sent['sentiment'].value_counts().idxmax()


if txt:
    st.write('Overall Sentiment:', run_sentiment_analysis(txt))
