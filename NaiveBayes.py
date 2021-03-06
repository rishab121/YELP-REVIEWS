from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

class NB:
    def __init__(self, feature_matrix_train, y_train):
        self.feature_matrix_train = feature_matrix_train
        self.y_train = feature_matrix_train
        self.clf = MultinomialNB()
        self.clf.fit(feature_matrix_train, y_train)

    def predictNB(self,feature_matrix_test,y_test):
        clf_predictions = self.clf.predict(feature_matrix_test)
        test_accuracy = str(metrics.accuracy_score(y_test, clf_predictions))
        print confusion_matrix(y_test, clf_predictions)
        print '\n'
        print classification_report(y_test, clf_predictions)

        return test_accuracy

    def predictRating(self,X_test):
        clf_predictions = self.clf.predict(X_test)
        return clf_predictions


