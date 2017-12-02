import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import sklearn as sk
import string
import re

reviews = pd.read_csv('./dataset/review.csv',chunksize = 10000)
def remove_punc(text):
	letters_only = re.sub('[^a-zA-Z]', ' ',text)
	words = letters_only.lower().split()
	return remove_stopwords(words)

def remove_stopwords(words):
	stopwords_eng = set(stopwords.words("english"))
	useful_words = [word for word in words if not word in stopwords_eng]
	return(useful_words)


def getFeatureVector():
	for min_reviews in reviews:
		w_reviews = min_reviews[(min_reviews['stars'] == 1) | (min_reviews['stars'] == 5)]

		reviews_text = w_reviews['text']
		reviews_star = w_reviews['stars']
		
		feature_matrix = sk.feature_extraction.text.CountVectorizer(analyzer = remove_punc).fit(reviews_text)
		reviews_text = feature_matrix.transform(reviews_text)
		return reviews_text


print getFeatureVector().shape





