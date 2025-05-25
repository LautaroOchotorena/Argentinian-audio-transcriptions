import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras
import keras_tuner as kt
from preprocessing import *
from model import Transformer
from config import max_target_len
import json

# load dataset
df = pd.read_csv('./data/df.csv')

# df['transcription'] = df['transcription'].apply(remove_accents)

# Split to df_train and df_val.
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(len(df) * 0.10)
df_train = df[split:]
df_val = df[:split]

train_dataset, valid_dataset = train_and_val_slice(df_train, df_val, batch_size=batch_size)

def build(hp):
    """Hyper tuner"""
    num_hid = hp.Int("num_hid", 200, 300, step=50)
    num_head = hp.Int("num_head", 2, 4)
    num_feed_forward = hp.Int("num_feed_forward", 250, 400, step=50)
    num_layers_enc = hp.Int("num_layers_enc", 4, 6, step=2)
    num_layers_dec = hp.Int("num_layers_dec", 2, 4)
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.2)

    model = Transformer(
        num_hid=num_hid,
        num_head=num_head,
        num_feed_forward=num_feed_forward,
        target_maxlen=max_target_len,
        num_layers_enc=num_layers_enc,
        num_layers_dec=num_layers_dec,
        num_classes=vocab_size,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        steps_per_epoch=len(train_dataset)
    )

    dummy_source = np.random.rand(batch_size, max_time_len, fft_length//2 + 1)
    dummy_target = np.random.randint(0, model.num_classes, size=(batch_size, max_target_len))

    model([dummy_source, dummy_target])

    loss_fn = keras.losses.CategoricalCrossentropy(
    from_logits=True,
    label_smoothing=0.1,
    )

    optimizer = keras.optimizers.Adam(model.scheduler_lr)
    model.compile(optimizer=optimizer, loss=loss_fn)
    return model

tuner = kt.Hyperband(
    build,
    objective="val_loss",
    max_epochs=15,
    factor=3,
    max_retries_per_trial=2,
    directory="tuner_logs",
    project_name="tuning"
)

tuner.search(
    train_dataset,
    validation_data=valid_dataset,
    epochs=15
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Saves the best hyperparameters
best_hps_dict = {k: best_hps.values[k] for k in best_hps.values}
with open("best_hyperparameters.json", "w") as f:
    json.dump(best_hps_dict, f, indent=4)