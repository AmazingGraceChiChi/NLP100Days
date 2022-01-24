# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 23:04:39 2022

@author: User
"""


'''
結巴默認的字典為簡體字點，在此我們使用預設字典針對繁體與簡體字串進行斷詞
'''
# 使用預設詞庫進行四種模式的斷詞
import jieba #載入結巴模組
jieba.set_dictionary('dict.txt.big')

input_traditional_str = '小明碩士畢業於國立臺灣大學，現在在日本東京大學進修深造' #定義須斷詞的繁體字串
input_simple_str = '小明硕士毕业于国立臺湾大学，现在在日本东京大学进修深造' #定義須斷詞的繁體字串

#使用paddle模式
'''
jieba.enable_paddle() #啟用paddle模式

trad_words = jieba.cut(input_traditional_str, use_paddle=True)
sim_words = jieba.cut(input_simple_str, use_paddle=True)

trad_paddle_output = []
sim_paddle_outupt = []

for word in trad_words:
    trad_paddle_output.append(word)
    
for word in sim_words:
    sim_paddle_outupt.append(word)

print(f'paddle模式段詞結果:')
print(trad_paddle_output)
print(sim_paddle_outupt)
print('\n')
'''

#使用全模式
trad_words = jieba.cut(input_traditional_str, cut_all=True)
sim_words = jieba.cut(input_simple_str, cut_all=True)


trad_all_output = [word for word in trad_words]
sim_all_output = [word for word in sim_words]


print(f'全模式段詞結果:')
print(trad_all_output)
print(sim_all_output)
print('\n')

#使用精確模式
trad_words = jieba.cut(input_traditional_str) #default is accurate mode
sim_words = jieba.cut(input_simple_str) #default is accurate mode

trad_acc_output = [word for word in trad_words]
sim_acc_output = [word for word in sim_words]

print(f'精確模式段詞結果:')
print(trad_acc_output)
print(sim_acc_output)
print('\n')

#搜尋引擎模式
trad_words = jieba.cut_for_search(input_traditional_str) 
sim_words = jieba.cut_for_search(input_simple_str) 

trad_search_output = [word for word in trad_words]
sim_search_output = [word for word in sim_words]

print(f'搜尋引擎模式段詞結果:')
print(trad_search_output)
print(sim_search_output)
print('\n')

# In[]
test_string = "他来到了网易杭研大厦" #此舉例為結巴官方repo的例子

#使用HMM
hmm_words = jieba.cut(test_string, HMM=True) #HMM 參數預設即為True
hmm_out = [word for word in hmm_words]

    
print(f'HMM output: {hmm_out}')
print('\n')

#不使用HMM
no_hmm_words = jieba.cut(test_string, HMM=False) #HMM 參數預設即為True
no_hmm_out = [word for word in no_hmm_words]

print(f'Non HMM output: {no_hmm_out}')
print('\n')

# In[]
# 動態調整字典
jieba.add_word('國立臺灣大學') #動態加入字詞
jieba.suggest_freq('國立臺灣大學', True) #動態調整詞頻

# In[]
import jieba.posseg as pseg

input_str = '小明碩士畢業於國立臺灣大學，現在在日本東京大學進修深造' #定義須斷詞的字串
words = pseg.cut(input_str)

for word, flag in words:
    print(word, flag)

# In[]
words = jieba.tokenize(input_str)

for tk in words:
    print(f'word:{tk[0]}, start:{tk[1]}, end:{tk[2]}')






















