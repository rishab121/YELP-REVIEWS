import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import sklearn as sk
import string
import re
from svm import SVM
from NaiveBayes import NB
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from rf import RF


reviews = pd.read_csv('./dataset/review.csv',nrows = 10000)
w_reviews = reviews[(reviews['stars'] == 1) | (reviews['stars'] == 5) | (reviews['stars'] == 3)]
reviews_text = w_reviews['text']
reviews_star = w_reviews['stars']

def remove_punc(text):
	letters_only = re.sub('[^a-zA-Z]', ' ',text)
	words = letters_only.lower().split()
	return remove_stopwords(words)

def remove_stopwords(words):
	stopwords_eng = set(stopwords.words("english"))
	useful_words = [word for word in words if not word in stopwords_eng]
	return(useful_words)


feature_matrix = sk.feature_extraction.text.CountVectorizer(analyzer = remove_punc).fit(reviews_text)
reviews_text_transformed = feature_matrix.transform(reviews_text)
X_train, X_test, y_train, y_test = train_test_split(reviews_text_transformed, reviews_star, test_size=0.3, random_state=101)


reviews['text length'] = reviews['text'].apply(len)
def graph():
    g = sns.FacetGrid(data=reviews, col='stars')
    g.map(plt.hist, 'text length', bins=50)
    plt.show()
    sns.boxplot(x='stars', y='text length', data=reviews)
    plt.show()


def svmFunction():
    svm = SVM(X_train,y_train)
    test_accuracy = svm.predictSvm(X_test,y_test)
    print "test accuracy of SVM is",test_accuracy
    print " Sample prediction of the rating by SVM for a Positive review"
    pos_review = w_reviews['text'][0]
    pos_review_transformed = feature_matrix.transform([pos_review])
    print svm.predictRating(pos_review_transformed)

    print " Sample prediction of the rating by SVM for a Negative review"
    neg_review = w_reviews['text'][16]
    neg_review_transformed = feature_matrix.transform([neg_review])
    print svm.predictRating(neg_review_transformed)

    print " Sample prediction of the rating by SVM for a Neutral review"
    neutral_review = w_reviews['text'][1]
    neutral_review_transformed = feature_matrix.transform([neutral_review])
    print svm.predictRating(neutral_review_transformed)


    our_review = "The food was good but the drinks was bad"
    print "our test for sample text :: ",our_review
    our_review_transformed = feature_matrix.transform([our_review])
    print "Rating of our review",svm.predictRating(our_review_transformed)


def nbFunction():
    nb = NB(X_train,y_train)
    test_accuracy = nb.predictNB(X_test,y_test)
    print "test accuracy of Naive Bayes is", test_accuracy
    print " Sample prediction of the rating by NB for a Positive review"
    pos_review = w_reviews['text'][0]
    pos_review_transformed = feature_matrix.transform([pos_review])
    print nb.predictRating(pos_review_transformed)

    print " Sample prediction of the rating by NB for a Negative review"
    neg_review = w_reviews['text'][16]
    neg_review_transformed = feature_matrix.transform([neg_review])
    print nb.predictRating(neg_review_transformed)

    print " Sample prediction of the rating by NB for a Neutral review"
    neutral_review = w_reviews['text'][1]
    neutral_review_transformed = feature_matrix.transform([neutral_review])
    print nb.predictRating(neutral_review_transformed)

    our_review = "The food was good but the drinks was bad"
    print "our test for sample text :: ", our_review
    our_review_transformed = feature_matrix.transform([our_review])
    print "Rating of our review", nb.predictRating(our_review_transformed)

def randomForest():
    rf = RF(X_train, y_train)
    test_accuracy = rf.predictRF(X_test, y_test)
    print "test accuracy of Naive Bayes is", test_accuracy
    print " Sample prediction of the rating by RF for a Positive review"
    pos_review = w_reviews['text'][0]
    pos_review_transformed = feature_matrix.transform([pos_review])
    print rf.predictRating(pos_review_transformed)

    print " Sample prediction of the rating by RF for a Negative review"
    neg_review = w_reviews['text'][16]
    neg_review_transformed = feature_matrix.transform([neg_review])
    print rf.predictRating(neg_review_transformed)

    print " Sample prediction of the rating by RF for a Neutral review"
    neutral_review = w_reviews['text'][1]
    neutral_review_transformed = feature_matrix.transform([neutral_review])
    print rf.predictRating(neutral_review_transformed)

    our_review = "Horrible food "
    print "our test for sample text :: ", our_review
    our_review_transformed = feature_matrix.transform([our_review])
    print "Rating of our review", rf.predictRating(our_review_transformed)

svmFunction()
nbFunction()
randomForest()


