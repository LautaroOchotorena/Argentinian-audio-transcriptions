import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import mlflow
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jiwer import wer, cer
from preprocessing import *
from config import (fft_length, rnn_units,
                    rnn_layers, default_initial_epoch, 
                    default_learning_rate, run_id, batch_size)
from mlflow.tracking import MlflowClient
from model import build_model

# Loads the trainning and validation dataset
train_dataset, validation_dataset = train_and_val_slice(df_train, df_val,
                                                        batch_size=batch_size)

# MLflow
# If a run_id is given then takes the last epoch and learning rate to continue
# the trainning
if run_id:
    print(f"Continuing the existing run: {run_id}")
    run = mlflow.start_run(run_id=run_id)

    client = MlflowClient()
    try:
        # Acces to the learning rate metric
        metrics = client.get_metric_history(run_id, "learning_rate")
        if metrics:
            # Takes the last value of the learning rate
            last_learning_rate = metrics[-1].value
            print(f"Last learning_rate found in MLflow: {last_learning_rate}")
            learning_rate = last_learning_rate
            # Last epoch that was trained
            initial_epoch = len(metrics)
            print(f"Last epoch trained: {initial_epoch}")
        else:
            # No metric was recorded
            print("No learning_rate found, using default values")
            learning_rate = default_learning_rate
            initial_epoch = default_initial_epoch
            print(f"First epoch to train: {default_initial_epoch}")

    except Exception as e:
        print(f"Error obtaining learning_rate: {e}. Using default values")
        learning_rate = default_learning_rate
        initial_epoch = default_initial_epoch
        print(f"First epoch to train: {default_initial_epoch}")
else:
    print("Creating a new run")
    run = mlflow.start_run(run_name="transcription_model")
    initial_epoch = default_initial_epoch
    learning_rate = default_learning_rate

# Builds the model
model = build_model(
    input_dim=fft_length//2 + 1,
    output_dim=char_to_num.vocabulary_size(),
    rnn_units=rnn_units,
    rnn_layers=rnn_layers,
    learning_rate=learning_rate
)
model.summary(line_length=110)

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

# A callback class to save
# metrics and outputs a few transcriptions after each epoch
class CallbackEval(keras.callbacks.Callback):
    """Displays a few batches of outputs after every epoch."""
    def __init__(self, dataset, valid_set=True, max_batches=50):
        super().__init__()
        self.valid_set = valid_set
        self.dataset = dataset
        self.max_batches = max_batches

    def on_epoch_end(self, epoch: int, logs=None):
        if not self.valid_set:
            # Each time it calls take it shuffles
            dataset = self.dataset.take(self.max_batches)
            # First saves the loss and val_loss
            logs = logs or {}
            if 'loss' in logs:
                mlflow.log_metric("loss", logs['loss'], step=epoch)
            if 'val_loss' in logs:
                mlflow.log_metric("val_loss", logs['val_loss'], step=epoch)
        else:
            dataset = self.dataset
        predictions = []
        targets = []
        for batch in dataset:
            X, y = batch
            batch_predictions = model.predict(X, verbose=0)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        cer_score = cer(targets, predictions)
        tag = "Validation Set" if self.valid_set else "Part of the Training Set"

        print(f"\n{'-' * 40} {tag} {'-' * 40}")
        print(f"Word Error Rate: {wer_score:.4f}")
        print(f"Character Error Rate: {cer_score:.4f}")
        print(f"{'-' * (len(tag) + 82)}")

        if self.valid_set:
            mlflow.log_metric("val_wer_score", wer_score, step=epoch)
            mlflow.log_metric("val_cer_score", cer_score, step=epoch)
            for i in np.random.randint(0, len(predictions), 2):
                print(f"Target    : {targets[i]}")
                print(f"Prediction: {predictions[i]}")
                print("-" * 100)
            
            # Store the learning rate as a metric
            lr = self.model.optimizer.learning_rate
            if callable(lr):
                lr = lr(self.model.optimizer.iterations).numpy()
            else:
                lr = lr.numpy()
            mlflow.log_metric("learning_rate", lr, step=epoch)
        else:
            mlflow.log_metric("wer_score", wer_score, step=epoch)
            mlflow.log_metric("cer_score", cer_score, step=epoch)

# Callback function to check transcriptions and metrics on the val set.
validation_callback = CallbackEval(validation_dataset, valid_set=True)
# Callback function to check metrics on the train set.
train_callback = CallbackEval(train_dataset, valid_set=False)

# Callback to plot and save metrics
class MetricsPlotCallback(keras.callbacks.Callback):
    def __init__(self, run_id: str, save_dir: str = "metric_images"):
        super().__init__()
        self.run_id = run_id
        self.save_dir = save_dir
        self.client = MlflowClient()
        # Creates the directory (if needed) to store the plots
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # === Retrieve metrics ===
        loss = self.client.get_metric_history(self.run_id, "loss")
        val_loss = self.client.get_metric_history(self.run_id, "val_loss")
        wer = self.client.get_metric_history(self.run_id, "wer_score")
        val_wer = self.client.get_metric_history(self.run_id, "val_wer_score")

        loss_vals = [m.value for m in sorted(loss, key=lambda x: x.step)]
        val_loss_vals = [m.value for m in sorted(val_loss, key=lambda x: x.step)]
        wer_vals = [m.value for m in sorted(wer, key=lambda x: x.step)]
        val_wer_vals = [m.value for m in sorted(val_wer, key=lambda x: x.step)]

        # === Loss plot ===
        plt.figure(figsize=(8, 5))
        plt.plot(loss_vals, label="loss", color="blue")
        plt.plot(val_loss_vals, label="val_loss", color="orange")
        plt.title("Loss vs Val_Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"loss_plot.png"))
        plt.close()

        # === WER plot ===
        plt.figure(figsize=(8, 5))
        plt.plot(wer_vals, label="wer", color="green")
        plt.plot(val_wer_vals, label="val_wer", color="red")
        plt.title("WER vs Val_WER")
        plt.xlabel("Epoch")
        plt.ylabel("Word Error Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"wer_plot.png"))
        plt.close()

# Define the number of epochs.
epochs = 100

# Callback to save the weights after each epoch
checkpoint_callback = ModelCheckpoint(
    filepath='checkpoints/epoch_{epoch:02d}.weights.h5',
    save_weights_only=True,
    save_freq='epoch',
    verbose=1
)

# If a intial_epoch was passed then it restores the weights from that epoch
if initial_epoch >0:
    model.load_weights(f'checkpoints/epoch_{initial_epoch:02d}.weights.h5')

# Reduce the learning rate if val_loss doesn't get better after a few epochs
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1)

with run:
    # If new run then stores the initial params
    if not run_id:
        mlflow.log_param("initial_epoch", initial_epoch)
        mlflow.log_param("model_type", "CTC")
        mlflow.log_param("input_dim", fft_length//2 + 1)
        mlflow.log_param("rnn_units", rnn_units)
        mlflow.log_param("rnn_layers", rnn_layers)
        run_id = run.info.run_id

    # Train the model
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[train_callback, validation_callback, checkpoint_callback,
                   lr_callback, MetricsPlotCallback(run_id=run_id)],
        initial_epoch=initial_epoch
    )