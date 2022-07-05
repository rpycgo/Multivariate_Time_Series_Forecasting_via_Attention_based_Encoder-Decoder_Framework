from ...config.config import model_config
from ..layers.attention import TemporalAttention

import tensorflow as tf
from tensorflow.keras.layers import Layer, Attention, Dense, LSTM, Lambda, RepeatVector


class Decoder(Layer):
    def __init__(self, config=model_config, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.config = config

        self.temporal_attention = TemporalAttention(name='temporal_attention')
        self.dense = Dense(units=1)
        self.lstms = [
            LSTM(units=config.decoder_lstm_units, return_state=True, return_sequences=True, name=f'decoder_lstm_{i}')
            for i in range(config.num_decoders)
            ]
        self.lstm = LSTM(units=config.decoder_lstm_units, return_sequences=True, name='decoder_lstm')

    def _apply_multiple_lstm(self, x):
        for idx, lstm in enumerate(self.lstms):
            if len(self.lstms) == 1:
                _, hidden_state, cell_state = lstm(x)
            elif len(self.lstms) >= 2:                        
                if idx == 0 or idx != (len(self.lstms)-1):
                    x, _, _ = lstm(x)                    
                else:
                    _, hidden_state, cell_state = lstm(x)
        
        return hidden_state, cell_state

    def call(self, inputs, encoder_output, training=None):
        _context_vector = tf.zeros((self.config.batch_size, 1, self.config.encoder_lstm_units)) # batch_size, 1, encoder_lstm_units

        context_vectors = tf.TensorArray(tf.float32, size=self.config.time_seq)
        for t in range(self.config.time_seq):
            x = Lambda(lambda x: inputs[:, t, :])(inputs)
            x = x[:, tf.newaxis, :]
            x = tf.concat([x, _context_vector], axis=-1)    # batch_size, 1, features + encoder_lstm_units
            x = self.dense(x)   # batch_size, 1, 1

            hidden_state, cell_state = self._apply_multiple_lstm(x)
            
            attention_weights = self.temporal_attention(hidden_state, cell_state, encoder_output)   # batch_size, time_seq, 1
            _context_vector = tf.matmul(attention_weights, encoder_output, transpose_a=True) # batch_size, 1, features
            context_vector = tf.concat([hidden_state[:, tf.newaxis, :], _context_vector], axis=-1) # batch_size, 1, features + encoder_lstm_units

            context_vectors = context_vectors.write(t, context_vector)

        context_vectors = tf.reshape(
            context_vectors.stack(), 
            shape=(-1, self.config.time_seq, context_vector.shape[-1])
            )   # batch_size, time_seq, features + encoder_lstm_units

        decoder_output = self.lstm(context_vectors) # batch_size, time_seq, decoder_lstm_units

        return decoder_output
