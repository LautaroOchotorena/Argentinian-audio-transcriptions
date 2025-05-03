import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
from jiwer import wer
import re
from preprocessing import *
from model import build_model
from config import rnn_units, rnn_layers, batch_size, default_learning_rate

# A utility function to decode the output of the network
def decode_batch_predictions(pred, greedy=True, beam_width=100, top_paths=1):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Greedy search or beam search apply
    if greedy==False and top_paths>1:
        results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=greedy, beam_width=beam_width, top_paths=top_paths)[0]
    else:
        results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=greedy, beam_width=beam_width, top_paths=top_paths)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for paths in zip(*results):  # Agrupa por muestra
        seen = set()
        unique_texts = []
        for path in paths:
            text = tf.strings.reduce_join(num_to_char(path)).numpy().decode("utf-8")
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)
        output_text.append(unique_texts)
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
    weights_folder = 'checkpoints'
    files = os.listdir(weights_folder)
    epoch_files = [f for f in files if re.match(r'epoch_\d+\.weights\.h5', f)]
    epochs = [int(re.search(r'epoch_(\d+)', f).group(1)) for f in epoch_files]
    latest_epoch = max(epochs)
    latest_weights_file = f"epoch_{latest_epoch:02d}.weights.h5"

    model.load_weights(os.path.join(weights_folder, latest_weights_file))
    return model

if __name__ == '__main__':
    train_dataset, validation_dataset = train_and_val_slice(df_train, df_val,
                                                        batch_size=batch_size)
    # Get the model
    model = load_model()

    # Predictions
    predictions = []
    targets = []
    for batch in validation_dataset:
        X, y = batch
        batch_predictions = model.predict(X)
        batch_predictions = decode_batch_predictions(batch_predictions)
        predictions.extend(batch_predictions)
        for label in y:
            label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
            targets.append(label)
    wer_score = wer(targets, predictions)
    print("-" * 100)
    print(f"Word Error Rate: {wer_score:.4f}")
    print("-" * 100)
    for i in np.random.randint(0, len(predictions), 5):
        print(f"Target    : {targets[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-" * 100)