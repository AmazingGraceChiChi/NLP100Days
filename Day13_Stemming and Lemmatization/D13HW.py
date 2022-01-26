# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 12:54:23 2022

@author: User
"""
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('wordnet')
## 創建stemmer
ps = PorterStemmer()

## 創建Lemmatizer
lemmatizer = WordNetLemmatizer() 

# In[]
print('Stemming amusing : {}'.format(ps.stem('amusing')))
print('lemmatization amusing : {}'.format(lemmatizer.lemmatize('amusing', pos = 'v')))

# In[]
# Define the sentence to be lemmatized
sentence = "The striped bats are hanging on their feet for best"

# Tokenize: Split the sentence into words
word_list = nltk.word_tokenize(sentence)
print(word_list)
#> ['The', 'striped', 'bats', 'are', 'hanging', 'on', 'their', 'feet', 'for', 'best']

stemming_output = ' '.join([ps.stem(w) for w in word_list])
print(stemming_output)
#> The striped bat are hanging on their foot for best

# In[]
# Define the sentence to be lemmatized
sentence = "The striped bats are hanging on their feet for best"

# Tokenize: Split the sentence into words
word_list = nltk.word_tokenize(sentence)
print(word_list)
#> ['The', 'striped', 'bats', 'are', 'hanging', 'on', 'their', 'feet', 'for', 'best']

# Lemmatize list of words and join
lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
print(lemmatized_output)
#> The striped bat are hanging on their foot for best

# In[]
# Lemmatize with POS Tag
from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """將pos_tag結果mapping到lemmatizer中pos的格式"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

word = 'using'
print(lemmatizer.lemmatize(word, get_wordnet_pos(word)))
