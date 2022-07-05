from ...config.config import model_config

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, RepeatVector


class TemporalAttention(Layer):
    def __init__(self, config=model_config, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.config = config
        self.v = Dense(1)
    
    def build(self, input_shape):
        self.w1 = Dense(input_shape[-1])
        self.w2 = Dense(input_shape[-1])
        self.repeat_vector = RepeatVector(self.config.time_seq)

    def call(self, hidden_state, cell_state, encoder_hidden_state, training=None):
        query = tf.concat([hidden_state, cell_state], axis=-1)
        query = self.repeat_vector(query)

        score = tf.nn.tanh(self.w1(encoder_hidden_state) + self.w2(query))
        score = self.v(score)

        attention_weights = tf.nn.softmax(score, axis=-1)

        return attention_weights
