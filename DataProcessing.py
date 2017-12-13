import pandas as pd
from nltk.corpus import stopwords
import sklearn as sk
import re
from svm import SVM
from NaiveBayes import NB
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from rf import RF
import random


reviews = pd.read_csv('./dataset/review.csv',nrows = 10000)
w_reviews = reviews[(reviews['stars'] == 1) | (reviews['stars'] == 5) | (reviews['stars'] == 3)] # data set
goodWords = [] # list of positive words created while training out naive bayes implementation
badWords = [] # list of negative words created while training out naive bayes
reviews_text = w_reviews['text'] # text of the reviews on which sentiment analysis will be performed
reviews_star = w_reviews['stars'] # label of the review

#Given : some string text
#Returns: list of the words in string without punctations and stopwords
def remove_punc(text):
	letters_only = re.sub('[^a-zA-Z]', ' ',text)
	words = letters_only.lower().split()
	return remove_stopwords(words)

def remove_stopwords(words):
	stopwords_eng = set(stopwords.words("english"))
	useful_words = [word for word in words if not word in stopwords_eng]
	return(useful_words)

# feature vector matrix for sk learn algos random forest,svm and naive bayes
feature_matrix = sk.feature_extraction.text.CountVectorizer(analyzer = remove_punc).fit(reviews_text)
reviews_text_transformed = feature_matrix.transform(reviews_text)

# spliting of the feature vector in training and testing
X_train, X_test, y_train, y_test = train_test_split(reviews_text_transformed, reviews_star, test_size=0.3, random_state=101)


# Data analysis
reviews['text length'] = reviews['text'].apply(len)
def graph():
    g = sns.FacetGrid(data=reviews, col='stars')
    g.map(plt.hist, 'text length', bins=50)
    plt.show()
    sns.boxplot(x='stars', y='text length', data=reviews)
    plt.show()


# Our implementation of the naive bayes algorithm
def naiveBayesTrain():
    bayesian_reviews_train = reviews[0:9000] # training set
    five_star_reviews = bayesian_reviews_train['text'][(bayesian_reviews_train['stars'] == 5)] # getting text of five star reviews
    one_star_reviews = bayesian_reviews_train['text'][(bayesian_reviews_train['stars'] == 1)] # getting text of one star reviews

    for text in five_star_reviews:
        for word in remove_punc(text):
            goodWords.append(word) # creating good words list

    for text in one_star_reviews:
        for word in remove_punc(text):
            badWords.append(word) # creating bad words list


def naiveBayesPredict(text):
    review_words = remove_punc(text) # clean review
    goodWordsCounter = 0
    badWordsCounter = 0
    length_of_review = len(review_words)
    for word in review_words:
        if word in goodWords:
            goodWordsCounter +=1
        if word in badWords:
            badWordsCounter +=1

    # calculate probability
    if length_of_review > 0:
        goodProb = float(goodWordsCounter) / length_of_review #conditional property of review being good
        badProb = float(badWordsCounter) / length_of_review # probabilty of review being bad

        if goodProb < badProb:
            return 1
        elif goodProb > badProb:
            return 5
        else:
            flip = random.randint(0,1)
            if flip == 0:
                return 1
            else:
                return 5


    else:
        return 0


def svmFunction():
    print "---------------------SVM--------------------------"
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
    print "--------------------Sk learn Naive Bayes Multinominal--------------------"
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
    print "--------------------Random Forest---------------------"
    rf = RF(X_train, y_train)
    test_accuracy = rf.predictRF(X_test, y_test)
    print "test accuracy of Random Forest is", test_accuracy
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


def testOurNb():
    print "--------------------Our Naive Bayes Binomial--------------------------"
    naiveBayesTrain()
    bayesian_reviews_test = reviews[6000:10000]
    five_star_reviews = bayesian_reviews_test['text'][(bayesian_reviews_test['stars'] == 5)]
    one_star_reviews = bayesian_reviews_test['text'][(bayesian_reviews_test['stars'] == 1)]
    five_star_correct_prediction = 0
    one_star_correct_prediction = 0
    pos_review = w_reviews['text'][0]
    neg_review = w_reviews['text'][16]
    print "prediction for pos review", naiveBayesPredict(pos_review)
    print "prediction for neg review", naiveBayesPredict(neg_review)
    for review in five_star_reviews:
        if naiveBayesPredict(review) == 5:
            five_star_correct_prediction += 1
    for review in one_star_reviews:
        if naiveBayesPredict(review) == 1:
            one_star_correct_prediction +=1

    five_correctly_predicted = float(five_star_correct_prediction) / len(five_star_reviews)
    one_correctly_predicted = float(one_star_correct_prediction) / len(one_star_reviews)
    print "positive review predicted with efficency ", 100 * five_correctly_predicted
    print "negative review predicted with efficency ", 100 * one_correctly_predicted



print "------------------------Testing the code....----------------------"
testOurNb() # our naive bayes test
randomForest() # random forest test
nbFunction() # sk learn naive bayes test
svmFunction() # svm sk learn test





