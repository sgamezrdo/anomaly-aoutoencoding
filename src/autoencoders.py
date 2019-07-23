from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

def get_simple_autoencoder(X_train, codingfactor):
    """"Sets up a base Keras auto encoder without hidden layers.

    Args:
        X_train: Training dataset - needs to get the input dimensionality
        codingfactor: Factor that will set up the code length (encoded layer dimensionality)

    Returns:
        autoencoder, encoder, decoder
    """
    # Calculate the input dimesionality
    encoding_dim = round(X_train.shape[1] / codingfactor)  #
    # Set up input layer
    input_layer = Input(shape=(X_train.shape[1],), name="input_layer")
    # encoded layer
    encoded_layer = Dense(encoding_dim, activation="relu", name="encoded_layer")(input_layer)
    # decoded layer
    decoded_layer = Dense(X_train.shape[1], activation="sigmoid", name="decoded_layer")(encoded_layer)
    # this model maps an input to its reconstruction
    autoencoder = Model(input_layer, decoded_layer)
    # this model maps an input to its encoded representation
    encoder = Model(input_layer, encoded_layer)
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))

    return autoencoder, encoder, decoder

def get_autoencoder(X_train, codingfactor):
    """"Sets up a Keras auto encoder with hidden layers.

    Args:
        X_train: Training dataset - needs to get the input dimensionality
        codingfactor: Factor that will set up the code length (encoded layer dimensionality)

    Returns:
        autoencoder, encoder, decoder
    """
    # Calculate the input dimesionality
    encoding_dim = round(X_train.shape[1] / codingfactor)  #
    # Set up input layer
    input_layer = Input(shape=(X_train.shape[1],), name="input_layer")
    # encoded hidden layer
    encoded_layer_hid = Dense(round((encoding_dim + X_train.shape[1])/2), activation="relu", name="encoded_hidden_layer")(input_layer)
    # encoded layer
    encoded_layer = Dense(encoding_dim, activation="relu", name="encoded_layer")(encoded_layer_hid)
    # decoded hidden layer
    decoded_layer_hid = Dense(round((encoding_dim + X_train.shape[1])/2), activation="relu", name="decoded_hidden_layer")(encoded_layer)
    # decoded layer
    decoded_layer = Dense(X_train.shape[1], activation="sigmoid", name="decoded_layer")(decoded_layer_hid)
    # this model maps an input to its reconstruction
    autoencoder = Model(input_layer, decoded_layer)
    # this model maps an input to its encoded representation
    encoder = Model(input_layer, encoded_layer)
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last tow layers of the autoencoder model
    decoder_layer, decoder_layer_hid = autoencoder.layers[-1], autoencoder.layers[-2]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(decoder_layer_hid(encoded_input)))

    return autoencoder, encoder, decoder

def decode(X, encoder, decoder):
    """"Returns the decoded data, result of passing through an autoencoder

    Args:
        X: Data to be encoded and then decoded
        encoder: Encoder model of an auntoencoder
        decoder: Decoder model of an autoencoder

    Returns:
        Decoded data
    """
    X_encoded = encoder.predict(X)
    X_decoded = decoder.predict(X_encoded)
    return X_decoded

def dist_decoding_err(X, encoder, decoder):
    dec_X = decode(X, encoder, decoder)
    err = np.apply_along_axis(np.linalg.norm, axis=1, arr=(X - dec_X))
    return err