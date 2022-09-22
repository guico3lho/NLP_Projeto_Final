import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from nltk import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords


class Split:
    pass


# train = df_copy[df_copy['Date'] < '20150101']
# test = df_copy[df_copy['Date'] > '20141231']

class Train(Split):
    def __init__(self, df):
        self.features = df.iloc[:, 3:28]
        self.y = df['Label']
        self.headlines = []
        self.corpus = []
        self.X = None

    def concat_headlines(self):
        for row in range(0, len(self.features.index)):
            self.headlines.append(' '.join(str(x) for x in self.features.iloc[row, 0:25]))
        return self.headlines

    @staticmethod
    def _normalizarString(text):

        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)

        return text

    def preProcessing(self):
        ps = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        for headline in self.headlines:
            text = self._normalizarString(headline)
            words = text.split()
            # words = [word for word in words if not word in set(stopwords.words('english'))]
            stem_words = [ps.stem(word) for word in words]
            # words = [lemmatizer.lemmatize(word) for word in stem_words]
            words = stem_words.copy()
            text = ' '.join(words)
            self.corpus.append(text)


class Test(Split):
    def __init__(self, df):
        self.features = df.iloc[:, 3:28]
        self.y = df['Label']
        self.headlines = []
        self.corpus = []
        self.X = None

    def concat_headlines(self):
        for row in range(0, len(self.features.index)):
            self.headlines.append(' '.join(str(x) for x in self.features.iloc[row, 0:25]))
        return self.headlines

    @staticmethod
    def _normalizarString(text):

        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)

        return text

    def preProcessing(self):
        ps = PorterStemmer()
        # lemmatizer = WordNetLemmatizer()
        for headline in self.headlines:
            text = self._normalizarString(headline)
            words = text.split()
            # words = [word for word in words if not word in set(stopwords.words('english'))]
            stem_words = [ps.stem(word) for word in words]
            words = stem_words.copy()
            # words = [lemmatizer.lemmatize(word) for word in stem_words]
            text = ' '.join(words)
            self.corpus.append(text)


class BagOfWords:
    def __init__(self):
        self.type = None
        self.X_train = None
        self.X_test = None

    def cv(self, train_corpus, test_corpus):
        sw = stopwords.words('english')
        self.type = CountVectorizer(stop_words=sw, ngram_range=(1, 1))

        train_matrix = self.type.fit_transform(train_corpus).toarray()
        self.X_train = pd.DataFrame(train_matrix, columns=self.type.get_feature_names_out())

        test_matrix = self.type.transform(test_corpus).toarray()
        self.X_test = pd.DataFrame(test_matrix, columns=self.type.get_feature_names_out())

        return self.X_train, self.X_test
        # return self.cv.get_feature_names()

    def tfidf(self, train_corpus, test_corpus):
        sw = stopwords.words('english')
        self.type = TfidfVectorizer(stop_words=sw, ngram_range=(1, 1))

        self.X_train = self.type.fit_transform(train_corpus).toarray()
        train_features = pd.DataFrame(self.X_train, columns=self.type.get_feature_names_out())

        self.X_test = self.type.transform(test_corpus).toarray()
        test_features = pd.DataFrame(self.X_test, columns=self.type.get_feature_names_out())


        return self.X_train, self.X_test


class Models:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.classifier = None
        self.y_pred = None
        self.cm = None
        self.score1 = None
        self.score2 = None
        self.score3 = None
        self.name = None

    def score(self):
        self.score1 = accuracy_score(self.y_test, self.y_pred)
        self.score2 = precision_score(self.y_test, self.y_pred)
        self.score3 = recall_score(self.y_test, self.y_pred)
        self.score4 = f1_score(self.y_test, self.y_pred)

        print("---- Scores ----")
        print("Accuracy score is: {}%".format(round(self.score1 * 100, 2)))
        print("Precision score is: {}".format(round(self.score2 * 100, 2)))
        print("Recall score is: {}".format(round(self.score3 * 100, 2)))
        print("F1 score is: {}".format(round(self.score4 * 100, 2)))
        print(classification_report(self.y_test, self.y_pred, labels=[0, 1]))

    def show_confusion_matrix(self):
        print("Confusion Matrix")
        plt.figure(figsize=(10, 7))

        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.title('Confusion Matrix')
        plt.show()

    def logistic_regression(self):
        print("Logistic Regression")
        self.name = 'Logistic Regression'
        self.classifier = LogisticRegression()
        self.classifier.fit(self.X_train, self.y_train)
        self.y_pred = self.classifier.predict(self.X_test)
        self.cm = confusion_matrix(self.y_test, self.y_pred)
        return self.classifier, self.y_pred, self.cm

    def naive_bayes(self):
        self.name = 'Naive Bayes'
        print("Naive Bayes")
        self.classifier = MultinomialNB()
        self.classifier.fit(self.X_train, self.y_train)
        self.y_pred = self.classifier.predict(self.X_test)
        self.cm = confusion_matrix(self.y_test, self.y_pred)
        return self.classifier, self.y_pred, self.cm

    def random_forest(self):
        self.name = 'Random Forest'
        print("Random Forest")
        self.classifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
        self.classifier.fit(self.X_train, self.y_train)
        self.y_pred = self.classifier.predict(self.X_test)
        self.cm = confusion_matrix(self.y_test, self.y_pred)
        return self.classifier, self.y_pred, self.cm

    def decision_tree(self):
        self.name = 'Decision Tree'
        print("Decision Tree")
        self.classifier = DecisionTreeClassifier(criterion='entropy')
        self.classifier.fit(self.X_train, self.y_train)
        self.y_pred = self.classifier.predict(self.X_test)
        self.cm = confusion_matrix(self.y_test, self.y_pred)
        return self.classifier, self.y_pred, self.cm

    def svm(self):
        self.name = 'SVM'
        print("SVM")
        self.classifier = SVC(kernel='linear', random_state=0)
        self.classifier.fit(self.X_train, self.y_train)
        self.y_pred = self.classifier.predict(self.X_test)
        self.cm = confusion_matrix(self.y_test, self.y_pred)
        return self.classifier, self.y_pred, self.cm

    def k_neighbors(self):
        self.name = 'K Neighbors'
        print("K Neighbors")
        self.classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        self.classifier.fit(self.X_train, self.y_train)
        self.y_pred = self.classifier.predict(self.X_test)
        self.cm = confusion_matrix(self.y_test, self.y_pred)
