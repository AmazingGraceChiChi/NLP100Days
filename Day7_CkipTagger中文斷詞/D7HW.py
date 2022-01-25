# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:34:20 2022

@author: User
"""
#導入所需libraries
#請import 1.可用於下載權重的library 2.建構自定義字典的library 3.斷詞, 詞性標注,與命名實體辨識libries
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER

sentence_list = [
    "傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。",
    "美國參議院針對今天總統布什所提名的勞工部長趙小蘭展開認可聽證會，預料她將會很順利通過參議院支持，成為該國有史以來第一位的華裔女性內閣成員。",
    "",
    "土地公有政策?？還是土地婆有政策。.",
    "… 你確定嗎… 不要再騙了……",
    "最多容納59,000個人,或5.9萬人,再多就不行了.這是環評的結論.",
    "科長說:1,坪數對人數為1:3。2,可以再增加。",
]

# In[]
#創建實例
ws = WS('./D07data/')
pos = POS('./D07data/')
ner = NER('./D07data/')

#斷詞
word_s = ws(sentence_list,
            sentence_segmentation=True,
            segment_delimiter_set={'?', '？', '!', '！', '。', ',','，', ';', ':', '、'})

print(f'斷詞輸出: {word_s}')
print('\n')

#詞性標注
word_p = pos(word_s)

print(f'詞性標注輸出: {word_p}')
print('\n')

#命名實體識別
word_n = ner(word_s, word_p)
print(f'命名實體識別輸出: {word_n}')

# In[]
#定義字典
word_to_weight = {
    "年前": 1,
}
dictionary = construct_dictionary(word_to_weight)

#帶入自定義字典進行斷詞

input_traditional_str = ["傅達仁今將執行安樂死，卻突然爆出自己20年前遭緯來體育台封殺，他不懂自己哪裡得罪到電視台。"]

word_sentence_list = ws(
    input_traditional_str,
    sentence_segmentation = True, # To consider delimiters
    segment_delimiter_set = {",", "。", ":", "?", "!", ";"}, # This is the defualt set of delimiters
    recommend_dictionary = dictionary)


print(word_sentence_list)


































