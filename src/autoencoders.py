from keras.layers import Input, Dense
from keras.models import Model

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
