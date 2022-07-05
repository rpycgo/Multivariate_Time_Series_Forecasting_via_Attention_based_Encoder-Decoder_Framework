from ...config.config import model_config
from ..layers.encoder import Encoder
from ..layers.decoder import Decoder

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


class MTSMFF(Model):
    def __init__(self, config=model_config, **kwargs):
        super(MTSMFF, self).__init__(**kwargs)
        self.config = config

        self.encoder= Encoder(config=config, name='encoder')
        self.decoder= Encoder(config=config, name='decoder')

        self.dense1 = Dense(units=1)
        self.dense2 = Dense(units=config.p)
        
    def call(self, inputs, training=None):
        encoder_output = self.encoder(inputs)   # batch_size, time_seq, features
        decoder_output = self.decoder(inputs, encoder_output)   # batch_size, time_seq, decoder_lstm_units

        _output = self.dense1(decoder_output)    # batch_size, time_seq, 1
        _output = tf.squeeze(_output)   # batch_size, time_seq
        output = self.dense2(_output)   # batch_size, p

        return output
