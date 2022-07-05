class model_config:
    batch_size = 96
    time_seq = 20
    p = 10
    epochs=0.3
    dropout_rate = 0.3

    # encoder
    num_encoders = 1
    encoder_lstm_units=100

    # decoder
    num_decoders = 1
    decoder_lstm_units=100

    merge_mode = 'sum'
