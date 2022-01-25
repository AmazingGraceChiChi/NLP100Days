# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 09:07:52 2022

@author: User
"""

from ckiptagger import data_utils, WS
data_utils.download_data_gdown("./")

# In[]
#建構斷詞
ws = WS("./data")

input_string = '小明碩士畢業於國立臺灣大學，現在在日本東京大學進修深造'
word_sentence_list = ws(
    input_string, 
    sentence_segmentation = True, # To consider delimiters
    segment_delimiter_set = {",", "。", ":", "?", "!", ";"}) # This is the defualt set of delimiters
print(word_sentence_list)

# In[]
from ckiptagger import POS

pos = POS("./data")
pos_sentence_list = pos(word_sentence_list)
print(pos_sentence_list)

# In[]
ner = NER("./data")

entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
print(entity_sentence_list)

# In[]
from ckiptagger import construct_dictionary

word_to_weight = {"日本東京大學": 1}
dictionary = construct_dictionary(word_to_weight)

# In[]
from ckiptagger import construct_dictionary

word_to_weight = {"日本東京大學": 1}
dictionary = construct_dictionary(word_to_weight)

# In[]
ws = WS("./data")
input_traditional_str = ['小明碩士畢業於國立臺灣大學，現在在日本東京大學進修深造']
word_sentence_list = ws(input_traditional_str, recommend_dictionary=dictionary)
print(word_sentence_list)























