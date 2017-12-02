import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import sklearn as sk
import string

reviews = pd.read_csv('review.csv')
#print reviews.shape
#print "head"
#print reviews.head()
#print "info"
#info = reviews.info()
#print  reviews.describe()
# very large dataset will consider for upto 10000 reviews
min_reviews = reviews[0:10000]
w_reviews = min_reviews[(min_reviews['stars'] == 1) | (min_reviews['stars'] == 5)]

#pd.DataFrame.to_csv(w_reviews)
reviews_text = w_reviews['text']
reviews_star = w_reviews['stars']
# have to remove stopwords and punctation in text

# Given: some String
# returns: List of the words in the string without punctations and stopwords
def remove_chars(text):
    text_list = [char for char in text if char not in string.punctuation] # will return list of char without puncs.
    text_list = ''.join(text_list) # will convert it into string
    return remove_stopwords(text_list) # to remove stopwords


def remove_stopwords(text_list):
    filtered_list = []  # my list of filtered words

    for word in text_list.split():
        if word.lower() not in stopwords.words('english'):
            filtered_list.append(word)

    return filtered_list



feature_matrix = sk.feature_extraction.text.CountVectorizer(analyzer = remove_chars).fit(reviews_text)
reviews_text = feature_matrix.transform(reviews_text)
print reviews_text



#print w_reviews.shape
