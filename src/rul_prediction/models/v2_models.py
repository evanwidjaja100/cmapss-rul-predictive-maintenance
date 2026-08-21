"""Methodology V2 sequence models: masking-aware LSTM / GRU / TCN.

V2 training includes left-padded short-history windows (shared history
builder), so the models consume TWO inputs: the padded window and a binary
mask (1 = observed timestep, 0 = padded). LSTM/GRU receive the mask through
the layer's native ``mask`` argument; the TCN zeroes padded timesteps via a
mask multiply (causal convs have no native mask - padding contributes only
zeros, and the same padded representation appears in training).
"""

from __future__ import annotations


def _optimizer(learning_rate: float, loss: str, clipnorm: float = 1.0):
    from tensorflow import keras

    # ponytail: optimizer clipnorm is behavior-driving; passed explicitly from config
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=float(clipnorm))
    return optimizer


def v2_lstm(window: int, n_features: int, units=(128, 64), dropout: float = 0.3,
            loss: str = "mse", learning_rate: float = 1e-3, seed: int = 42, clipnorm: float = 1.0):
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    inputs = keras.Input(shape=(window, n_features))
    mask_in = keras.Input(shape=(window,), name="mask")
    x = keras.layers.LSTM(units[0], return_sequences=True)(inputs, mask=mask_in)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.LSTM(units[1])(x, mask=mask_in)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model([inputs, mask_in], outputs)
    model.compile(optimizer=_optimizer(learning_rate, loss, clipnorm), loss=loss)
    return model


def v2_gru(window: int, n_features: int, units=(128, 64), dropout: float = 0.3,
           loss: str = "mse", learning_rate: float = 1e-3, seed: int = 42, clipnorm: float = 1.0):
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    inputs = keras.Input(shape=(window, n_features))
    mask_in = keras.Input(shape=(window,), name="mask")
    x = keras.layers.GRU(units[0], return_sequences=True)(inputs, mask=mask_in)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.GRU(units[1])(x, mask=mask_in)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model([inputs, mask_in], outputs)
    model.compile(optimizer=_optimizer(learning_rate, loss, clipnorm), loss=loss)
    return model


def v2_tcn(window: int, n_features: int, filters: int = 64, dilations=(1, 2, 4, 8),
           dropout: float = 0.2, loss: str = "mse", learning_rate: float = 1e-3,
           seed: int = 42, clipnorm: float = 1.0):
    from tensorflow import keras

    keras.utils.set_random_seed(seed)

    def residual_block(x, dilation: int):
        shortcut = x
        skip = keras.layers.Conv1D(filters, 1)(shortcut)
        y = keras.layers.Conv1D(filters, 3, padding="causal", dilation_rate=dilation)(x)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.ReLU()(y)
        y = keras.layers.Dropout(dropout)(y)
        y = keras.layers.Conv1D(filters, 3, padding="causal", dilation_rate=dilation)(y)
        y = keras.layers.BatchNormalization()(y)
        return keras.layers.ReLU()(keras.layers.Add()([y, skip]))

    inputs = keras.Input(shape=(window, n_features))
    mask_in = keras.Input(shape=(window,), name="mask")
    masked = keras.layers.Multiply()([inputs, keras.layers.Reshape((window, 1))(mask_in)])
    x = keras.layers.Conv1D(filters, 3, padding="causal")(masked)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    for dilation in dilations:
        x = residual_block(x, dilation)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model([inputs, mask_in], outputs)
    model.compile(optimizer=_optimizer(learning_rate, loss, clipnorm), loss=loss)
    return model