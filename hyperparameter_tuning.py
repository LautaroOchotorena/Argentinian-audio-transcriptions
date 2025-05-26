import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import keras
from tensorflow.keras import layers
import keras_tuner as kt
from preprocessing import *
from config import fft_length
from model import CTCLoss
import json

# load dataset
df = pd.read_csv('./data/df.csv')

#df['transcription'] = df['transcription'].apply(remove_accents)

# df = df.sample(n=1000, random_state=42)

# Split to df_train and df_val.
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(len(df) * 0.10)
df_train = df[split:]
df_val = df[:split]

input_dim = fft_length//2 + 1
output_dim=char_to_num.vocabulary_size()

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        """Hyper tuner"""
        rnn_layers = hp.Int("rnn_layers", 4, 5)
        dropout_rate = hp.Choice("dropout_rate", [0.3, 0.5])
        rnn_units = hp.Int("rnn_units", 384, 512, step=64)
        learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        # Model's input
        input_spectrogram = layers.Input((None, input_dim), name="input")

        # Expand the dimension to use 2D CNN.
        # The model requeries to be in form (time_steps, bins, 1)
        x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
        # Convolution layer 1
        x = layers.Conv2D(
            filters=32,
            kernel_size=[11, 41],
            strides=[2, 2],
            padding="same",
            use_bias=False,
            name="conv_1",
        )(x)
        x = layers.BatchNormalization(name="conv_1_bn")(x)
        x = layers.ReLU(name="conv_1_relu")(x)
        # Convolution layer 2
        x = layers.Conv2D(
            filters=32,
            kernel_size=[11, 21],
            strides=[1, 2],
            padding="same",
            use_bias=False,
            name="conv_2",
        )(x)
        x = layers.BatchNormalization(name="conv_2_bn")(x)
        x = layers.ReLU(name="conv_2_relu")(x)
        # Reshape the resulted volume to feed the RNNs layers
        x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
        # RNN layers
        for i in range(1, rnn_layers + 1):
            recurrent = layers.GRU(
                units=rnn_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                use_bias=True,
                return_sequences=True,
                reset_after=True,
                name=f"gru_{i}",
            )
            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if i < rnn_layers:
                x = layers.Dropout(rate=dropout_rate)(x)
        # Dense layer
        x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
        x = layers.ReLU(name="dense_1_relu")(x)
        x = layers.Dropout(rate=dropout_rate)(x)
        # Classification layer
        output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
        # Model
        model = keras.Model(input_spectrogram, output)
        # Optimizer
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
        # Compile the model and return
        model.compile(optimizer=opt, loss=CTCLoss)

        return model

tuner = kt.Hyperband(
    MyHyperModel(),
    objective="val_loss",
    max_epochs=15,
    factor=3,
    max_retries_per_trial=2,
    directory="tuner_logs",
    project_name="tuning"
)

train_subdataset, valid_subdataset = train_and_val_slice(df_train, df_val, batch_size=batch_size)

tuner.search(
    train_subdataset,
    validation_data=valid_subdataset,
    epochs=15
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Saves the best hyperparameters
best_hps_dict = {k: best_hps.values[k] for k in best_hps.values}
with open("best_hyperparameters.json", "w") as f:
    json.dump(best_hps_dict, f, indent=4)