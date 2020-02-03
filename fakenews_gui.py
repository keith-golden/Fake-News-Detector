#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 22:33:41 2018

This applies a Naive Bayes algorithm on a training set of news articles labeled as FAKE or REAL to be used for predicting other articles.
The commented out code can be used to assist in inspecting the model.

The first half of the file contains the machine learning code. The second half contains the GUI.

Edited: ???, Added GUI interface for detection of text input
Edited: 4/7/19, Added website article scraper

@author: Keith Golden
"""

# Import the necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import tkinter as tk


##########################################################################
# Build the DataFrame, training sets, and test sets.
# Maximizing training set size to improve performance. Performance was already tested using a large test size and proved to be proficient. 

# Build the DataFrame
df = pd.read_csv("news_train.txt", index_col=0)

# Print the head of df
#print(df.head())

# Create a series to store the labels: y
y = df.label

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.05, random_state=53)


#########################################################################
# TfidfVectorizer for Text Classification
# tfidf_vectorizer  .fit() method learns the vocabulary of words, assigns a unique number to each word
# .transform() creates matrices where each row is vector for a news article, each column is a unique word in the vocabulary, each value is the tf-idf frequency.


# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

# Fit and transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print 10 features (These are 10 words alplhabetically sorted from X_train. First hundreds are numerical words, hence pulling from 9990)
#   print(tfidf_vectorizer.get_feature_names()[9990:10000])

# Print the first 5 vectors of the tfidf training data
#   print(tfidf_train.A[:5])


##########################################################################
# (Optional) Inspecting the vectors 
"""
This takes the tfidf_train matrix above and converts it to a DataFrame for optional inspection. 
The word tokens/feature names are the column names. The dfs values are the word vectors.
"""
   
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

# Print the head of tfidf_df
#   print(tfidf_df.head())

############################################################################
#Training the "Fake News" model 
"""At this point we have not trained the model. We have taken articles, vectorized them, and turned
those vectors of words into two matrices.

Here we will apply a Naive Bayes model to those matrices """

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB(alpha=0.1)

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

###############################################################################
# Inspecting the model 
""" This shows the 30 words most heavily weighted 'FAKE' or 'REAL'"""

# Get the class labels: class_labels
# class_labels = nb_classifier.classes_

# Extract the features (The unique words in the vocabulary of all articles)
# feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
# feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 30 feat_with_weights entries (Words most associated with FAKE)
# print(class_labels[0], feat_with_weights[0:30])

# Print the second class label and the bottom 30 feat_with_weights entries (Words most associated with REAL)
# print(class_labels[1], feat_with_weights[-30:])

################################################################################
# Building the GUI

window = tk.Tk()
frame = tk.Frame(window)
frame.pack()

# Add title
window.title("Fake News Detector")

# Entry column
entry = tk.Entry(frame)
entry.pack(side="top")

# Tests the article being provided by user
def run_detection():
    user_article_input = str(entry.get())
    #user_article_input = "https://www.nytimes.com/2020/02/02/us/politics/iowa-economy-2020.html?action=click&module=Top%20Stories&pgtype=Homepage"

    #Check if url
    if "http" in user_article_input[0:9] or "www." in user_article_input[0:9]:
        from fakenews_scraper import ArticleScraper
        scraps = ArticleScraper(user_article_input)
        user_article_input = scraps.article
    
    user_article = pd.Series(user_article_input)

    # Transform the test data: tfidf_test 
    tfidf_user_article = tfidf_vectorizer.transform(user_article)

    # Create the predicted tags: pred_CV
    user_article_pred_tfidf = nb_classifier.predict(tfidf_user_article)
    verdict = str(user_article_pred_tfidf)
    verdict = verdict.strip("[']")
    
    #bttn.configure(text="This article is " + verdict + user_article_input + "!")
    bttn.configure(text="This article is " + verdict + "!")
    
    #print(user_article_input)
    #print(user_article)
    #print(user_article_pred_tfidf)

bttn = tk.Button(window, text = "Execute", command=run_detection)
bttn.pack(side="bottom")

bottom = tk.Label(frame, text="\n==================================================\nEnter a url to a news article above, or enter text to an article.\n Press the button below to run the detection.\n ==================================================\n\n\n\n\n\nBuilt by Keith Golden\n")
bottom.pack(side='bottom')
window.mainloop()







