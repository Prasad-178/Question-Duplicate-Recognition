import numpy as np
import pandas as pd

from preprocessing import Preprocess
from train import Train
from model import SiameseModel

from config import max_len, embedding_dim

data = pd.read_csv('./data/train.csv')

preprocessor = Preprocess()
trainer = Train()
modeller = SiameseModel()

data, padded_sequences_1, padded_sequences_2, tokenizer, embedding_matrix =  preprocessor(data)

my_model = modeller(shape=(padded_sequences_1.shape[1],), vocab_size=len(tokenizer.word_index)+1, max_len=max_len, embedding_dim=embedding_dim, embedding_matrix=embedding_matrix)

history = trainer((padded_sequences_1, padded_sequences_2), 0.8, my_model, data['is_duplicate'])
