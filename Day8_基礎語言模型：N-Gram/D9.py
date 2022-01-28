# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 21:55:41 2022

@author: User
"""

#讀取文本資料

import requests

books = {'Pride and Prejudice': '1342',
         'Huckleberry Fin': '76',
         'Sherlock Holmes': '1661'}

book = books['Pride and Prejudice']


url_template = f'https://www.gutenberg.org/cache/epub/{book}/pg{book}.txt'
response = requests.get(url_template)
txt = response.text

#檢查文本
print(len(txt), ',', txt[:50] , '...')

# In[]
import re

words = re.split(r'[^A-Za-z]+', txt.lower())
words = [x for x in words if x != ''] # 移除空字串

print(len(words))

# In[]
# unigram
unigram_frequecy = dict()

for word in words:
    unigram_frequecy[word] = unigram_frequecy.get(word, 0) + 1
    
# 根據詞頻排序, 並轉換為(word, count)格式
unigram_frequecy = sorted(unigram_frequecy.items(), key=lambda word_count: word_count[1], reverse=True)

# 查看詞頻前10的字詞
print(unigram_frequecy[:10])

# In[]
# bigram
bigram_frequency = dict()

for i in range(0, len(words)-1):
    bigram_frequency[tuple(words[i:i+2])] = bigram_frequency.get(tuple(words[i:i+2]), 0) + 1
    
# 根據詞頻排序, 並轉換為(word, count)格式
bigram_frequency = sorted(bigram_frequency.items(), key=lambda words_count: words_count[1], reverse=True)

# 查看詞頻前10的字詞
bigram_frequency[:10]

# In[]
def do_bigram_prediction(bigram_freq, start_word, num_words):
    #定義起始字
    pred_words = [start_word]
    word = start_word
    for i in range(num_words):
        # 找尋下一個字
        word = pred_words[i]
        word = next((word_pairs[1] for (word_pairs, count) in bigram_freq if word_pairs[0] == word), None)
        
        if not word:
            break
        else:
            pred_words.append(word)
    return pred_words

# In[]
import random

# 隨機選取起始字
start_word = random.choice(words)
print(f'初始字: {start_word}')

# 使用選取的起始字預測接下來的字詞(共10個字)
pred_words = do_bigram_prediction(bigram_frequency, start_word, 10)

# 顯示預測結果
print(f"預測句子: {' '.join(pred_words)}")

# In[]
# 權重選取
def weighted_choice(choices):
   total = sum(w for c, w in choices)
   r = random.uniform(0, total)
   upto = 0
   for c, w in choices:
      if upto + w > r:
         return c
      upto += w
    
def do_bigram_weighted_prediction(bigram_freq, start_word, num_words):
    pred_words = [start_word]
    # word = start_word
    for i in range(num_words):
        # 選取所有符合條件的2gram
        word = pred_words[i]
        words_candidates = [word_pairs_count for word_pairs_count in bigram_freq if word_pairs_count[0][0] == word]
        if not words_candidates:
            break
        else:
            #根據加權機率選取可能的字詞
            pred_words.append(weighted_choice(words_candidates)[1])
            
    return pred_words

# In[]
start_word = 'of'
print(f'初始字: {start_word}')

pred_words = do_bigram_weighted_prediction(bigram_frequency, start_word, 10)
print(f"預測句子: {' '.join(pred_words)}")

# In[]
def generateNgram(N):
    gram_frequency = dict()
    
    # 避免N值過大，導致記憶體崩潰問題(先設定N < 100)
    assert N > 0 and N < 100
    
    # 建立N-gram的頻率字典
    for i in range(len(words)-(N-1)):
        gram_frequency[tuple(words[i:i+N])] = gram_frequency.get(tuple(words[i:i+N]), 0) + 1

    # 根據詞頻排序, 並轉換為(word, count)格式
    gram_frequency = sorted(gram_frequency.items(), key=lambda words_count: words_count[1], reverse=True)
    
    return gram_frequency

# In[]
# 建立Trigram
trigram_frequency = generateNgram(3)

# 查看詞頻前10的字詞
trigram_frequency[:10]

# In[]
#建立N-gram預測function
def do_ngram_weighted_prediction(gram_freq, start_word, num_words):
    pred_words = [start_word]
    # word = start_word
    for i in range(num_words):
        # 選取所有符合條件
        word = pred_words[i]
        words_candidates = [word_pairs_count for word_pairs_count in gram_freq if word_pairs_count[0][0] == word]
        
        if not words_candidates:
            break
        else:
            #根據加權機率選取可能的字詞
            pred_words.append(weighted_choice(words_candidates)[1])
            
    return pred_words

# In[]
start_word = 'of'
print(f'初始字: {start_word}')

pred_words = do_ngram_weighted_prediction(trigram_frequency, start_word, 10)
print(f"預測推薦字詞: {' '.join(pred_words)}")

# In[]
import collections
from nltk import ngrams

#使用NLTK API搭建Bigram
bigram_frequency = ngrams(words, n=2)

#使用collectins套件計算詞頻
bigram_frequency = collections.Counter(bigram_frequency)
print(bigram_frequency[:10])

