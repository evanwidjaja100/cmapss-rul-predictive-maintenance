"""Temporal Convolutional Network (TCN) for RUL forecasting.

Causal convolutions + residual dilated blocks. Receptive field with kernel 3
and dilations (1, 2, 4, 8): 1 + 2 * 2 * (1+2+4+8) = 61 timesteps >= the 30-step
window, so every past cycle influences the prediction. Uses exactly the same
preprocessing / splits / metrics as the LSTM and GRU baselines.
"""

from __future__ import annotations

DEFAULT_DILATIONS = (1, 2, 4, 8)


def _residual_block(x, filters: int, dilation: int, dropout: float):
    from tensorflow import keras

    shortcut = x
    skip = keras.layers.Conv1D(filters, 1)(shortcut)
    y = keras.layers.Conv1D(filters, 3, padding="causal", dilation_rate=dilation)(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.ReLU()(y)
    y = keras.layers.Dropout(dropout)(y)
    y = keras.layers.Conv1D(filters, 3, padding="causal", dilation_rate=dilation)(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Add()([y, skip])
    return keras.layers.ReLU()(y)


def tcn_model(sequence_length: int, n_features: int, filters: int = 64,
              dilations=DEFAULT_DILATIONS, dropout: float = 0.2,
              loss: str = "mse", learning_rate: float = 1e-3, seed: int = 42):
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    inputs = keras.Input(shape=(sequence_length, n_features))
    x = keras.layers.Conv1D(filters, 3, padding="causal")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    for dilation in dilations:
        x = _residual_block(x, filters, dilation, dropout)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss)
    return model