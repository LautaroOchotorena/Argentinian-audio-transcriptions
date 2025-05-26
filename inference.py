import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
from jiwer import wer, cer
import re
from model import build_model
from config import (characters, rnn_units, rnn_layers, batch_size,
                    fft_length, default_learning_rate)
import keras

# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# A utility function to decode the output of the network
def decode_batch_predictions(pred, greedy=True, beam_width=100, top_paths=1):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Greedy search or beam search apply
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=greedy, beam_width=beam_width, top_paths=top_paths)[0]

    if top_paths > 1:
        # Only stores the results that are different
        output_text = []
        for paths in zip(*results):
            seen = set()
            unique_texts = []
            for path in paths:
                text = tf.strings.reduce_join(num_to_char(path)).numpy().decode("utf-8")
                if text not in seen:
                    seen.add(text)
                    unique_texts.append(text)
            output_text.append(unique_texts)
        return output_text
    else:
        # Iterate over the results and get back the text
        output_text = []
        for result in results[0]:
            result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
            output_text.append(result)
        return output_text

def load_model():
    model = build_model(
        input_dim=fft_length//2 + 1,
        output_dim=char_to_num.vocabulary_size(),
        rnn_units=rnn_units,
        rnn_layers=rnn_layers,
        learning_rate=default_learning_rate
    )

    # Load the weights from the last checkpoint
    checkpoint_dir = 'checkpoints'
    files = os.listdir(checkpoint_dir)
    epoch_files = [f for f in files if re.fullmatch(r'ckpt-(\d+)\.data-00000-of-00001', f)]
    epochs = [int(re.search(r'ckpt-(\d+)', f).group(1)) for f in epoch_files]
    latest_epoch = max(epochs)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
    checkpoint.restore(f"{checkpoint_dir}/ckpt-{latest_epoch}")

    return model

if __name__ == '__main__':
    from preprocessing import *
    train_dataset, validation_dataset = train_and_val_slice(df_train, df_val,
                                                        batch_size=batch_size)
    # Get the model
    model = load_model()

    # Predictions
    predictions = []
    targets = []
    for batch in validation_dataset:
        X, y = batch
        batch_predictions = model.predict(X, verbose=0)
        batch_predictions = decode_batch_predictions(batch_predictions)
        predictions.extend(batch_predictions)
        for label in y:
            label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            targets.append(label)
    wer_score = wer(targets, predictions)
    cer_score = cer(targets, predictions)
    print("-" * 100)
    print(f"Word Error Rate: {wer_score:.4f}")
    print(f"Character Error Rate: {cer_score:.4f}")
    print("-" * 100)
    for i in np.random.randint(0, len(predictions), 5):
        print(f"Target    : {targets[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-" * 100)