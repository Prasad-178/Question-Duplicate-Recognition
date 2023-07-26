from typing import Any
import numpy as np
import pandas as pd
import nltk
import os

from nltk.corpus import stopwords
from collections import defaultdict
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from nltk.corpus import stopwords

from config import embedding_dim

glove_dir = './data'

class Preprocess:
    def __call__(self, data, stemming=False):
        data = self.removeStopwords(data=data)
        
        tokenizer = self.fit_tokenizer(data=data)
        
        word2ind = self.get_pretrained_embeddings()
        
        embedding_matrix = self.get_embedding_matrix(tokenizer, word2ind)
        
        padded_sequences_1, padded_sequences_2 = self.padding(tokenizer, data)
        
        return data, padded_sequences_1, padded_sequences_2, tokenizer, embedding_matrix
        
            
    def removeStopwords(self, data):
        stop_words = set(stopwords.words('english'))
        
        def func(sentence):
            return " ".join([word for word in sentence.split(' ') if word not in stop_words])
        
        data['question1_clean'] = data['question1'].apply(lambda x: str(x).lower())
        data['question2_clean'] = data['question2'].apply(lambda x: str(x).lower())
        
        data['question1_clean'] = data['question1_clean'].apply(lambda sentence: func(sentence))
        data['question2_clean'] = data['question2_clean'].apply(lambda sentence: func(sentence))
        
        data['text'] = data[['question1_clean', 'question2_clean']].apply(lambda x: str(x[0]) + " " + str(x[1]), axis=1)
        
        return data        
            
    def fit_tokenizer(self, data):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data['text'].values)
        
        return tokenizer
    
    def padding(self, tokenizer: Tokenizer, data):
        sequences1 = tokenizer.texts_to_sequences(data['question1_clean'])
        sequences2 = tokenizer.texts_to_sequences(data['question2_clean'])
        
        padded_sequences_1 = pad_sequences(sequences1, padding='post', maxlen=30)
        padded_sequences_2 = pad_sequences(sequences2, padding='post', maxlen=30)
        
        return padded_sequences_1, padded_sequences_2
    
    def get_pretrained_embeddings(self):
        word2ind = {}
        f = open(os.path.join(glove_dir, 'glove.6b.100d.txt'), encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word2ind[word] = coefs
        f.close()
        
        return word2ind
    
    def get_embedding_matrix(self, tokenizer: Tokenizer, word2ind: dict):
        vocab_size = len(tokenizer.word_index)+1
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        
        for word, i in tokenizer.word_index.items():
            if word in word2ind:
                embedding_vector = word2ind.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                embedding_matrix[i] = np.zeros(100)
                
        return embedding_matrix
            
