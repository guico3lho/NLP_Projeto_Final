
import pandas as pd
from sklearn.model_selection import train_test_split

from classes import Train, Test, BagOfWords, Models
def main():
    # %% Loading the dataset
    df_raw = pd.read_csv("Datasets/Stock Headlines.csv", encoding='ISO-8859-1')
    df = df_raw.copy()
    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    df_train,df_test = df[df['Date'] < '20150101'], df[df['Date'] > '20141231']
    # df_train,df_test = train_test_split(df,test_size=0.1,random_state=42)
    train = Train(df_train)
    test = Test(df_test)

    train.concat_headlines()
    test.concat_headlines()

    train.preProcessing()
    test.preProcessing()
    ...

    print("Using CountVectorizer")
    bow_cv = BagOfWords()
    train.X,test.X = bow_cv.cv(train.corpus, test.corpus)

    model = Models(train.X, test.X, train.y, test.y)
    model.naive_bayes()
    model.score()
    model.show_confusion_matrix()
    model.logistic_regression()
    model.score()
    model.random_forest()
    model.score()
    model.decision_tree()
    model.score()
    model.svm()
    model.score()
    model.k_neighbors()
    model.score()

    print("Using tfidf")
    bow_tfidf = BagOfWords()
    train.X,test.X = bow_tfidf.tfidf(train.corpus, test.corpus)

    model = Models(train.X, test.X, train.y, test.y)
    model.naive_bayes()
    model.score()
    model.show_confusion_matrix()
    model.logistic_regression()
    model.score()
    model.random_forest()
    model.score()
    model.decision_tree()
    model.score()
    model.svm()
    model.score()
    model.k_neighbors()
    model.score()
    6
if __name__ == '__main__':
    main()

