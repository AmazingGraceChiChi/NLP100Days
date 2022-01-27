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
#dataset=pd.read_csv('movie_feedback.csv',encoding = 'Big5',names=['feedback', 'label'] )
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




























