import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
from jiwer import wer
import re
from model import build_model
from config import (characters, batch_size, max_target_len,
                    num_hid, num_head, num_feed_forward,
                    num_layers_enc, num_layers_dec, default_learning_rate)
import keras

# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=["<", ">"] + characters,
                                        oov_token="",
                                        pad_to_max_tokens=True,
                                        max_tokens=max_target_len)
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
            vocabulary=char_to_num.get_vocabulary(),
            oov_token="", invert=True)

# A utility function to decode the output of the network
def decode_batch_predictions(results, greedy=True, beam_width=100, top_paths=1):
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        # Stop where the ">" representation char is found
        end_token_idx = tf.where(result == 2)
        if tf.size(end_token_idx).numpy() > 0:
            first_index = end_token_idx[0][0]
            result = result[:first_index]
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

def load_model():
    model = build_model(num_hid=num_hid,
        num_head=num_head,
        num_feed_forward=num_feed_forward,
        target_maxlen=max_target_len,
        num_layers_enc=num_layers_enc,
        num_layers_dec=num_layers_dec,
        num_classes=vocab_size + 1,
        learning_rate=default_learning_rate)

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
    batch_size= 10
    train_dataset, validation_dataset = train_and_val_slice(df_train, df_val,
                                                        batch_size=batch_size)
    # Get the model
    model = load_model()

    # Predictions
    predictions = []
    targets = []
    for batch in validation_dataset:
        X, y = batch
        batch_predictions = model.predict(X, target_start_token_idx=1)[:, 1:]
        batch_predictions = decode_batch_predictions(batch_predictions)
        predictions.extend(batch_predictions)
        batch_y = decode_batch_predictions(y[:, 1:])
        targets.extend(batch_y)
    wer_score = wer(targets, predictions)
    print("-" * 100)
    print(f"Word Error Rate: {wer_score:.4f}")
    print("-" * 100)
    for i in np.random.randint(0, len(predictions), 5):
        print(f"Target    : {targets[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-" * 100)