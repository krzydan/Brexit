import getopt
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import sys
from sklearn.model_selection import train_test_split

def read_txt(path):
    try:
        data = pd.read_csv(path, sep="\t", header=None)
    except FileNotFoundError:
        print("File "+ path+" not found")
        sys.exit(2)
    data.columns = ["class", "text"]

    data.dropna(inplace=True)

    data['text'] = [entry.lower() for entry in data['text']]


    data['text'] = [word_tokenize(entry) for entry in data['text']]


    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index, entry in enumerate(data['text']):

        Final_words = []

        word_Lemmatized = WordNetLemmatizer()

        for word, tag in pos_tag(entry):

            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                Final_words.append(word_Final)

        data.loc[index, 'text_final'] = str(Final_words)
    return data

def txt_representation(mode, data, data_test, n):
    if mode == "bag":
        X = bag(data, data_test, n)
        return X
    else:
        X = tfidf(data, data_test, n)
        return X

def bag(data, data_test, n):
    cnt = CountVectorizer(ngram_range=(1, n))
    cnt.fit(data['text_final'])
    X = cnt.transform(data['text_final'])
    X_test = cnt.transform(data_test['text_final'])
    return X,X_test

def tfidf(data, data_test,n):
    Tfidf_vect = TfidfVectorizer(max_features=5000, ngram_range= (1, n))
    Tfidf_vect.fit(data['text_final'])
    if type(data_test)==type(data):
        X_Tfidf_test = Tfidf_vect.transform(data_test['text_final'])
    else:
        X_Tfidf_test =''
    X_Tfidf = Tfidf_vect.transform(data['text_final'])

    return X_Tfidf,X_Tfidf_test

def classifier(type):
    if type == "naive":
        clf = naive_bayes.MultinomialNB()
    elif type == "knn":
        clf = KNeighborsClassifier(n_neighbors=17, weights='distance', metric='euclidean')
    elif type == "svm":
        clf = svm.SVC(C=1, kernel='linear', degree=3, gamma='auto')
    elif type == "ensemble":
        clf_naive = classifier("naive")
        clf_knn = classifier("knn")
        clf_svm = classifier("svm")
        clf = VotingClassifier(estimators=[('svm', clf_knn), ('svm2', clf_svm), ('svm3', clf_naive)])
    else:
        print("wrong classifier type")
        sys.exit(2)

    return clf

def ensemble(clf,clf2):
    enClassfier = VotingClassifier(estimators=[('gb', clf), ('svm', clf2)])
    return enClassfier


def main():
    learnset = ''
    testset = ''
    classif = ''
    try:
        opts, args = getopt.getopt(sys.argv[2:], "ht:c:", ["test=", "clf="])
    except getopt.GetoptError:
        print('__main__.py <learnset> -t <testset> -c <classifier>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('__main__.py <learnset> -t <testset> -c <classifier>')
            sys.exit()
        elif opt in ("-t", "--test"):
            testset = arg
        elif opt in ("-c", "--clf"):
            classif = arg
    print(classif)
    learnset= sys.argv[1]
    data = read_txt(learnset)
    if classif == '':
        classif='ensemble'
    if testset!= '' :
        data_test = read_txt(testset)
    else:
        data_test =''
    X, X_Test = txt_representation("tf-idf", data, data_test, 1)
    y = data['class']


    k_fold = KFold(n_splits=10, shuffle=True, random_state=None)
    clf = classifier(classif)
    i = 0
    scores = []
    cv_scores = cross_val_score(clf, X, y, cv=k_fold, n_jobs=1)
    print('cv_scores mean:{:.4f}'.format(np.mean(cv_scores)))



    if testset!='':
        y_test = data_test['class']
        clf.fit(X,y)
        predictions = clf.predict(X)
        print("Train set Accuracy Score ->{:.4f} ".format(accuracy_score(predictions, y)))
        predictions = clf.predict(X_Test)
        print("Test set Accuracy Score ->{:.4f} ".format(accuracy_score(predictions, y_test)))




if __name__ == '__main__':
    main()
