"""LSTM RUL baseline (leakage-safe: validation partition is engine-disjoint)."""

from __future__ import annotations


def lstm_model(sequence_length: int, n_features: int, loss: str = "mse",
               learning_rate: float = 1e-3, seed: int = 42):
    """Input -> LSTM 128 -> DO 0.3 -> LSTM 64 -> DO 0.3 -> Dense 32 -> Dense 1."""
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    inputs = keras.Input(shape=(sequence_length, n_features))
    x = keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(64)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss)
    return model