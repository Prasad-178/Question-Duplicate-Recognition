import tensorflow as tf
from tensorflow.python import keras

from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers import Input, Dense, Flatten, Concatenate, Multiply, Dropout, Subtract, Embedding, LSTM, Lambda, BatchNormalization, Bidirectional, concatenate

from eval import Eval

class SiameseModel:
    def __call__(self, shape, vocab_size, max_len, embedding_dim, embedding_matrix):
        input_1 = Input(shape=shape)
        input_2 = Input(shape=shape)
        
        word_embedding_1 = Embedding(input_dim=vocab_size, weights=[embedding_matrix], output_dim=embedding_dim, input_length=max_len, trainable=False)(input_1)
        word_embedding_2 = Embedding(input_dim=vocab_size, weights=[embedding_matrix], output_dim=embedding_dim, input_length=max_len, trainable=False)(input_2)
        
        lstm_1 = LSTM(128, return_sequences=True)(word_embedding_1)
        
        lstm_2 = LSTM(128, return_sequences=True)(word_embedding_2)
        
        # vector_1 = Dense(128, activation="relu")(lstm_1)
        # vector_1 = Dropout(0.1)(vector_1)
        
        # vector_2 = Dense(128, activation="relu")(lstm_2)
        # vector_2 = Dropout(0.1)(vector_2)
        
        # x3 = Subtract()([vector_1, vector_2])
        # x3 = Multiply()([x3, x3])
        
        # x1_ = Multiply()([vector_1, vector_1])
        # x2_ = Multiply()([vector_2, vector_2])
        # x4 = Subtract()([x1_, x2_])
        
        # x5 = Lambda(Eval.cosine_similarity, output_shape=Eval.cosine_similarity_shape)([lstm_1, lstm_2])
        
        # concat = Concatenate(axis=-1)([x5,x4,x3])
        
        concat = concatenate([lstm_1, lstm_2])
        
        merged = BatchNormalization()(concat)
        merged = Dropout(0.25)(merged)
        
        merged = Dense(128, activation="relu")(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(0.25)(merged)
        
        # x = Dense(128, activation="relu")(x5)
        # x = Dropout(0.01)(x)
        # x = BatchNormalization()(x)
        
        output = Dense(1, activation="sigmoid")(merged)
        
        model = Model(inputs=[input_1, input_2], outputs=output)
        
        model.compile(loss="binary_crossentropy", 
                      metrics=["accuracy"], 
                      optimizer='adam'
                )
        
        print(model.summary())
        
        return model
        
