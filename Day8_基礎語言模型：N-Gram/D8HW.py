# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 12:58:11 2022

@author: User
"""

import numpy as np
import pandas as pd
words = ['i', 'want', 'to', 'eat', 'chinese', 'food', 'lunch', 'spend']
word_cnts = np.array([2533, 927, 2417, 746, 158, 1093, 341, 278]).reshape(1, -1)
df_word_cnts = pd.DataFrame(word_cnts, columns=words)

# In[]
# 記錄當前字與前一個字詞的存在頻率
bigram_word_cnts = [[5, 827, 0, 9, 0, 0, 0, 2], [2, 0, 608, 1, 6, 6, 5, 1], [2, 0, 4, 686, 2, 0, 6, 211],
                    [0, 0, 2, 0, 16, 2, 42, 0], [1, 0, 0, 0, 0, 82, 1, 0], [15, 0, 15, 0, 1, 4, 0, 0],
                    [2, 0, 0, 0, 0, 1, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0]]

df_bigram_word_cnts = pd.DataFrame(bigram_word_cnts, columns=words, index=words)

# In[]
df_bigram_prob = df_bigram_word_cnts.copy()
for word in words:
    df_bigram_prob.loc[word, :] = df_bigram_word_cnts.loc[word, :] / df_word_cnts.loc[0, word]


















