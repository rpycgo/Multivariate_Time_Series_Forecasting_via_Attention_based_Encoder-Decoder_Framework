from ...config.config import model_config

import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Bidirectional, Concatenate


class Encoder(Layer):
    def __init__(self, config=model_config, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.config = config

        self.bi_lstms = [
            Bidirectional(
                LSTM(units=config.encoder_lstm_units, return_sequences=True),
                merge_mode=config.merge_mode
                ) for _ in range(config.num_encoders)
                ]

    def call(self, inputs, training=None):
        x = inputs

        for bi_lstm in self.bi_lstms:
            x = bi_lstm(x)
        
        return x
