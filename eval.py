import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.models import Model
import pandas as pd
from pandas import DataFrame

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so stopping training!")
            self.model.stop_training = True
        if logs.get('accuracy') is not None and logs.get('accuracy') < 0.01:
            print("\nVery low accuracy, something is wrong with the model!")
            self.model.stop_training = True


class Eval:
    def euclidean_distance(a, b):
        return np.linalg.norm(a-b)

    def cosine_similarity(vectors):
        x, y = vectors
        x = K.l2_normalize(x, axis=-1)
        y = K.l2_normalize(y, axis=-1)
        
        return -K.mean(x*y, axis=-1, keepdims=True)
    
    def cosine_similarity_shape(shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)
    
class Test:
    def __call__(model: Model, test_data: pd.DataFrame):
        model = tf.keras.models.load_model('final_model')
        
        test_data = pd.read_csv('./data/test.csv')
        
        scores = model.evaluate((test_data['question1'], test_data['question1']), test_data['is_duplicate'])
        
        print(f'test accuracy is {scores[1]}')