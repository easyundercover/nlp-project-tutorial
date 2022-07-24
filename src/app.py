#Import libraries
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#Import dataset
url = "https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv"
df_raw = pd.read_csv(url)

#Clean dataset:
df_int = df_raw.copy()
df_int = df_int.drop_duplicates().reset_index(drop = True)

#Defining functions for data cleaning:
def protocol(text):
    return re.sub(r'(https://www|https://)', '', text)

def punct(text):
    return re.sub('[^a-zA-Z]', ' ', text)

def char(text):
    return re.sub('(\\d|\\W)+',' ', text)

def space(text):
    return re.sub(' +', ' ', text)

df_int['clean_url'] = df_int['url'].apply(protocol).apply(char).apply(space).apply(punct)
df_int['is_spam'] = df_int['is_spam'].apply(lambda x: 1 if x == True else 0)
df = df_int.copy()

#Modelling
vectorizer = CountVectorizer().fit_transform(df['clean_url'])

X = df['clean_url']
y = df['is_spam']

X_train, X_test, y_train, y_test = train_test_split(vectorizer, y, stratify = y, random_state = 25)

classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)


