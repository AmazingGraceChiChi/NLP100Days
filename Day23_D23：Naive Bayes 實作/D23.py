# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 00:48:44 2022

@author: User
"""

import numpy as np
import pandas as pd

data = pd.read_csv(r'Tweets.csv')
data = data[['text','airline_sentiment']]
# In[]                   
label_to_index = {"negative": 0, "neutral": 1, "positive": 2}
			
## 去除開頭航空名稱 ex. @VirginAmerica
X = data['text'].apply(lambda x: ' '.join(x.split(' ')[1:])).values
## 將negative, neutral, postive 轉換為 0,1,2
Y = data['airline_sentiment'].apply(lambda x:label_to_index[x]).values

# In[]
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
import re
import nltk
nltk.download('stopwords')
# Lemmatize with POS Tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

## 創建Lemmatizer
lemmatizer = WordNetLemmatizer()
def get_wordnet_pos(word):
	"""將pos_tag結果mapping到lemmatizer中pos的格式"""
	tag = nltk.pos_tag([word])[0][1][0].upper()
	tag_dict = {"J": wordnet.ADJ,
							"N": wordnet.NOUN,
							"V": wordnet.VERB,
							"R": wordnet. ADV}
	return tag_dict.get(tag, wordnet.NOUN)
def clean_content(X):
	# remove non-alphabet characters
	X_clean = [re.sub('[^a-zA-Z]',' ', str(x)).lower() for x in x]
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

# In[]
from sklearn.feature_extraction.text import CountVectorizer
#max_features是要建造幾個column，會按造字出現的頻率高低去篩選，3600並沒有特別含義(筆者測試幾次最佳效果)
#大家可以自己嘗試不同數值或不加入限制
cv = CountVectorizer(max_features = 3600)
X_T = cv.fit_transform(X).toarray()

# 有 14640 個樣本，每個樣本用3600維表示
X_T.shape

from sklearn.model_selection import train_test_split
# random_state 是為了讓各位學員得到相同的結果，平時可以移除
X_train, X_test, y_train, y_test = train_test_split(X_T, Y, test_size = 0.2, random_state = 0)

# In[]
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

clf_M = MultinomialNB()
clf_M.fit(X_train, y_train)

# In[]
# 二元特徵
X_train[X_train > 1] = 1
X_test[X_test > 1] = 1

clf_B = BernoulliNB()
clf_B.fit(X_train, y_train)

print('Trainset Accuracy: {}'.format(clf_B.score(X_train, y_train)))
print('Testset Accuracy: {}'.format(clf_B.score(X_test, y_test)))

output_ = clf_B.predict(X_test)
cm_output = confusion_matrix(y_test, output_)
