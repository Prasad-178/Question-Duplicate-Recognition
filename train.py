import tensorflow as tf
import os

from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from config import checkpoint_dir

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)

# from tensorflow.python.framework.config import set_memory_growth
# tf.compat.v1.disable_v2_behavior()
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.99:
            print("\nReached 99% accuracy so stopping training!")
            self.model.stop_training = True
        if logs.get('accuracy') is not None and logs.get('accuracy') < 0.01:
            print("\nVery low accuracy, something is wrong with the model!")
            self.model.stop_training = True

class Train:
    def __call__(self, data, split, model, y_values):
        pad_1_train, pad_2_train, pad_1_val, pad_2_val, y_train, y_val = self.split_train_val(data, split, y_values)
        
        history = self.train_model(model, pad_1_train, pad_2_train, pad_1_val, pad_2_val, y_train, y_val)
        print(history)
        
        return history
        
        
    def split_train_val(self, data, split, y_values):
        padded_sequences_1, padded_sequences_2 = data
        num_train = int(len(padded_sequences_1) * split * 0.25)
        
        pad_1_train = padded_sequences_1[:num_train]
        pad_2_train = padded_sequences_2[:num_train]
        y_train = y_values[:num_train]
        
        pad_1_val = padded_sequences_1[num_train:int(len(padded_sequences_1)/4)]
        pad_2_val = padded_sequences_2[num_train:int(len(padded_sequences_1)/4)]
        y_val = y_values[num_train:int(len(padded_sequences_1)/4)]
        
        return pad_1_train, pad_2_train, pad_1_val, pad_2_val, y_train, y_val
        

    def train_model(self, model: Model, pad_1_train, pad_2_train, pad_1_val, pad_2_val, y_train, y_val):
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        callbacks = myCallback()
        # early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        model_checkpoint = ModelCheckpoint(checkpoint_dir, save_best_only=True, save_weights_only=False)
        
        history = model.fit(
            [pad_1_train, pad_2_train], 
            y_train, 
            epochs=20,
            batch_size=64,
            validation_data=([pad_1_val, pad_2_val], y_val),
            callbacks=[callbacks, model_checkpoint]
        )
        
        model.save('final_model')
        
        return history