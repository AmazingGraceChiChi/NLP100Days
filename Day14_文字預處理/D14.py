# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 20:39:32 2022

@author: User
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
#tsv是指用tab分開字元的檔案
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

# In[]
import re 
review = re.sub('[^a-zA-Z]',' ',dataset['Review'][0])

# In[]
#把全部變成小寫
review = review.lower()

# In[]
review = nltk.word_tokenize(review)

# In[]
# 中文使用結巴
import jieba
jieba.set_dictionary('dict.txt')

review_ = '哇！我好喜歡這個地方'
cut_result = jieba.cut(review_, cut_all=False, HMM=False)
print("output: {}".format('|'.join(cut_result)))

# In[]
nltk.download('stopwords')
review=[word for word in review if not word in set(stopwords.words('english'))]

# In[]
# source:https://github.com/tomlinNTUB/Machine-Learning
with open('停用詞-繁體中文.txt','r') as file:
    stop_words = file.readlines()
stop_words = [word.strip('\n') for word in stop_words]

practice_sentence = ['哈哈','!','現在','好想','睡覺','啊']
practice_sentence = [word for word in practice_sentence if not word in set(stop_words)]
print('practice_sentence after removeing stopwords : {}'.format(practice_sentence))

# In[]
# Stemming: 詞幹提取
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review]

# In[]
# 練習清理所有的句子
# dataset=pd.read_csv('movie_feedback.csv',encoding = 'Big5',names=['feedback', 'label'] )
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)

corpus=[]
row=len(dataset)
for i in range(0,row):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    ## 這裡先不用stopwords 因為 review中很多反定詞會被移掉 如isn't good, 會變成 good
    review = [ps.stem(word) for word in review]
    review = ' '.join(review)
    corpus.append(review)

# In[]
from collections import Counter
import nltk
## 從整個corpus中取出所有的單詞
whole_words = []
for sentence in corpus:
    for words in nltk.word_tokenize(sentence):
        whole_words.append(words)
    
## 取出出現頻率top_k的單詞
top_k = 1000
top_k_words = []
for item in Counter(whole_words).most_common(top_k):
    top_k_words.append(item[0])

# In[]
# 以第一個句子為例
remove_low_frequency_word = ' '.join([word for word in nltk.word_tokenize(corpus[0]) if word in set(top_k_words)])

# In[]
# 轉bag of words
from sklearn.feature_extraction.text import CountVectorizer
#Creating bag of word model
#tokenization(符號化)
#max_features是要建造幾個column，會按造字出現的高低去篩選
cv = CountVectorizer(max_features=1000)
#toarray是建造matrixs
#X現在為sparsity就是很多零的matrix
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values











































