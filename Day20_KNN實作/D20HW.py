# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 15:39:22 2022

@author: User
"""

# In[]
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import codecs
import re

# In[]
dataset = pd.read_csv(r'KNN_datasets.csv', encoding = 'latin-1')
all_data = []

for content, label in dataset[['v2','v1']].values:
    if label == 'spam':
        label = 1
    else:
        label = 0
    all_data.append([content, label])
all_data = np.array(all_data)

# In[]
X = all_data[:,0]
Y = all_data[:,1].astype(np.uint8)

# In[]
# 文字預處理
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords

import nltk

nltk.download('stopwords')

# Lemmatize with POS Tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 

# In[]
## 創建Lemmatizer
lemmatizer = WordNetLemmatizer() 
def get_wordnet_pos(word):
    """將pos_tag結果mapping到lemmatizer中pos的格式"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def clean_content(X):
    # remove non-alphabet characters
    X_clean = [re.sub('[^a-zA-Z]',' ', x).lower() for x in X]
    # tokenize
    X_word_tokenize = [nltk.word_tokenize(x) for x in X_clean]
    # stopwords_lemmatizer
    X_stopwords_lemmatizer = []
    stop_words = set(stopwords.words('english'))
    for content in X_word_tokenize:
        content_clean = []
        for word in content:
            if word not in stop_words:
                word = lemmatizer.lemmatize(word, get_wordnet_pos(word))
                content_clean.append(word)
        X_stopwords_lemmatizer.append(content_clean)
    
    X_output = [' '.join(x) for x in X_stopwords_lemmatizer]
    
    return X_output
X = clean_content(X)

# In[]
from sklearn.feature_extraction.text import CountVectorizer
#max_features是要建造幾個column，會按造字出現的高低去篩選 
cv=CountVectorizer(max_features = 2000)
X=cv.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

print('Trainset Accuracy: {}'.format(classifier.score(X_train, y_train)))
print('Testset Accuracy: {}'.format(classifier.score(X_test, y_test)))

# In[]
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

#n-jobs=-1，是指cpu全開
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
n_neighbors  = [5, 10, 25, 50, 100] ## 可自行嘗試不同K值
for k in n_neighbors:
    classifier = KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p = 2)
    # cv = 10 代表切成10等分
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10,n_jobs=-1)
    
    print('設置K值:{}'.format(k))
    print('Average Accuracy: {}'.format(accuracies.mean()))
    print('Accuracy STD: {}'.format(accuracies.std()))

