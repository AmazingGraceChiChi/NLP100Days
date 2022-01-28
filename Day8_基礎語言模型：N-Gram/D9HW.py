# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 08:58:38 2022

@author: User
"""

import json
import re
from collections import Counter, namedtuple

with open('./WebNews.json', 'r', encoding='utf-8') as f:
    news_data = json.load(f)

# In[]
#取出新聞內文
corpus_list = list(d['detailcontent'] for d in news_data)

#去除HTML Tags與標點符號(只保留英文、數字、中文)
corpus_list = list(''.join(re.findall(r'^<.*?>$|[\u4E00-\u9FA50-9]', article)) for article in corpus_list)

# In[]
def ngram(documents, N=2):
    
    #建立儲存預測字, 所有ngram詞頻字典, 所有字詞(分母)
    ngram_prediction = dict()
    total_grams = list()
    words = list()
    Word = namedtuple('Word', ['word', 'prob']) #使用namedtuple來儲存預測字詞與對應機率

    for doc in documents:
        # 在每個文章錢加上起始(<start>)與結束符號(<end>)
        split_words = ['<s>'] + list(doc) + ['</s>']
        # 計算分子
        [total_grams.append(tuple(split_words[i:i+N])) for i in range(len(split_words)-N+1)]
        # 計算分母
        [words.append(tuple(split_words[i:i+N-1])) for i in range(len(split_words)-N+2)]
    
    #計算分子詞頻
    total_word_counter = Counter(total_grams)
    #計算分母詞頻
    word_counter = Counter(words)
    
    #計算所有N-gram預測字詞的機率
    for key in total_word_counter:
        # print(key)
        word = ''.join(key[:N-1])
        if word not in ngram_prediction:
            ngram_prediction.update({word: set()})
            
        next_word_prob = total_word_counter[key]/word_counter[key[:N-1]]
        w = Word(key[-1], f'{next_word_prob}')
        ngram_prediction[word].add(w)
        
    return ngram_prediction

# In[]
#建立bigram模型，並將預測的機率按照大小排列
four_gram_pred = ngram(corpus_list, N=4)
for word, pred in four_gram_pred.items():
    four_gram_pred[word] = sorted(pred, key=lambda x: x.prob, reverse=True)
    
    
# In[]
#給定字詞，使用ngram預測下一個字的機率(顯示top 10)
text = '鄭文燦'
next_words = list(four_gram_pred[text])[:10]
for next_word in next_words:
    print('next word: {}, probability: {}'.format(next_word.word, next_word.prob))

























