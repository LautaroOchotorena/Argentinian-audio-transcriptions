import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import mlflow
import keras
import tensorflow.keras.backend as K
import matplotlib
matplotlib.use("Agg")
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from jiwer import wer, cer
from preprocessing import *
from inference import decode_batch_predictions
from config import (num_hid, num_head, num_feed_forward,
                    num_layers_enc, num_layers_dec, default_initial_epoch, 
                    default_learning_rate, dropout_rate,
                    run_id, batch_size)
from mlflow.tracking import MlflowClient
from model import build_model

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
            # Last epoch that was trained
            initial_epoch = len(metrics)
            print(f"Last epoch trained: {initial_epoch}")
        else:
            initial_epoch = default_initial_epoch
            print(f"First epoch to train: {default_initial_epoch}")

    except Exception as e:
        print(f"Error obtaining learning_rate: {e}. Using default values")
        initial_epoch = default_initial_epoch
        print(f"First epoch to train: {default_initial_epoch}")
else:
    print("Creating a new run")
    run = mlflow.start_run(run_name="transcription_model")
    initial_epoch = default_initial_epoch

# Builds the model
model = build_model(num_hid=num_hid,
        num_head=num_head,
        num_feed_forward=num_feed_forward,
        target_maxlen=max_target_len,
        num_layers_enc=num_layers_enc,
        num_layers_dec=num_layers_dec,
        num_classes=vocab_size,
        learning_rate=default_learning_rate,
        dropout_rate=dropout_rate,
        steps_per_epoch=len(train_dataset))

# Dummy inputs to create the summary
# It builds the model, necessary to then restore weights
dummy_source = np.random.rand(batch_size, max_time_len, fft_length//2 + 1)
dummy_target = np.random.randint(0, model.num_classes, size=(batch_size, max_target_len))

model([dummy_source, dummy_target])
model.summary()

# A callback class to save
# metrics and outputs a few transcriptions after each epoch
class CallbackEval(keras.callbacks.Callback):
    """Displays a few batches of outputs after every epoch."""
    def __init__(self, dataset, valid_set=True, max_batches=int(360/batch_size)):
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
                mlflow.log_metric("loss", logs['loss'], step=epoch+1)
            if 'val_loss' in logs:
                mlflow.log_metric("val_loss", logs['val_loss'], step=epoch+1)
        else:
            dataset = self.dataset
            # Store the learning rate as a metric
            lr = self.model.optimizer.learning_rate
            lr = float(K.get_value(lr))
            mlflow.log_metric("learning_rate", lr, step=epoch+1)
        
        # Calculate wer and cer only every 5 epochs
        if (epoch + 1) % 5 != 0 and (epoch + 1) != 1:
            return
        
        predictions = []
        targets = []
        for batch in dataset:
            X, y = batch
            batch_predictions = model.predict(X, target_start_token_idx=2, target_end_token_idx=3)[:, 1:]
            batch_predictions = decode_batch_predictions(batch_predictions, target_end_token_idx=3)
            predictions.extend(batch_predictions)
            batch_y = decode_batch_predictions(y[:, 1:])
            targets.extend(batch_y)
        wer_score = wer(targets, predictions)
        cer_score = cer(targets, predictions)
        tag = "Validation Set" if self.valid_set else "Part of the Training Set"

        print(f"\n{'-' * 40} {tag} {'-' * 40}")
        print(f"Word Error Rate: {wer_score:.4f}")
        print(f"Character Error Rate: {cer_score:.4f}")
        print(f"{'-' * (len(tag) + 82)}")

        if self.valid_set:
            mlflow.log_metric("val_wer_score", wer_score, step=epoch+1)
            mlflow.log_metric("val_cer_score", cer_score, step=epoch+1)
            for i in np.random.randint(0, len(predictions), 2):
                print(f"Target    : {targets[i]}")
                print(f"Prediction: {predictions[i]}")
                print("-" * 100)
        else:
            mlflow.log_metric("wer_score", wer_score, step=epoch+1)
            mlflow.log_metric("cer_score", cer_score, step=epoch+1)

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
        cer = self.client.get_metric_history(self.run_id, "cer_score")
        val_cer = self.client.get_metric_history(self.run_id, "val_cer_score")

        loss_vals = [(m.step, m.value) for m in sorted(loss, key=lambda x: x.step)]
        val_loss_vals = [(m.step, m.value) for m in sorted(val_loss, key=lambda x: x.step)]
        wer_vals = [(m.step, m.value) for m in sorted(wer, key=lambda x: x.step)]
        val_wer_vals = [(m.step, m.value) for m in sorted(val_wer, key=lambda x: x.step)]
        cer_vals = [(m.step, m.value) for m in sorted(cer, key=lambda x: x.step)]
        val_cer_vals = [(m.step, m.value) for m in sorted(val_cer, key=lambda x: x.step)]

        # === Loss plot ===
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(*zip(*loss_vals), label="loss", color="blue")
        ax.plot(*zip(*val_loss_vals), label="val_loss", color="orange")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title("Loss vs Val_Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, f"loss_plot.png"))
        plt.close(fig)

        # === WER plot ===
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(*zip(*wer_vals), label="wer", color="blue")
        ax.plot(*zip(*val_wer_vals), label="val_wer", color="orange")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Integer ticks
        ax.set_title("WER vs Val_WER")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Word Error Rate")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, f"wer_plot.png"))
        plt.close(fig)

        # === CER plot ===
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(*zip(*cer_vals), label="cer", color="blue")
        ax.plot(*zip(*val_cer_vals), label="val_cer", color="orange")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # Integer ticks
        ax.set_title("CER vs Val_CER")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Character Error Rate")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, f"cer_plot.png"))
        plt.close(fig)

checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)

# Saves and only keeps the max_to_keep files
class save_weights_and_optimizer_state(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, max_to_keep=15):
        super().__init__()
        self.manager = tf.train.CheckpointManager(
            checkpoint, 
            directory=checkpoint_dir,
            max_to_keep=max_to_keep
        )

    def on_epoch_end(self, epoch, logs=None):
        # Save the weights and the optimizer state
        save_path = self.manager.save()
        print(f'\nModel weights and optimizer state saved at: {save_path}')

checkpoint_dir = 'checkpoints'
# If a intial_epoch was passed then it restores the weights
# and the optimizer state from that epoch
if initial_epoch > 0:
    checkpoint.restore(checkpoint_dir + f'/ckpt-{initial_epoch}')
    print(f"Checkpoint loaded")

class MLFlowEarlyStopping_and_reduce_lr_cap(EarlyStopping):
    def __init__(self, run_id=None, monitor='val_loss',
                 early_stop_patience=6, min_delta=0.0,
                 reduce_lr_factor=0.5,
                 reduce_lr_patience=3,
                 verbose=1, **kwargs):
        super().__init__(patience=early_stop_patience, min_delta=min_delta,
                         verbose=verbose, **kwargs)
        self.run_id = run_id
        self.best_value = np.inf

        self.monitor = monitor
        # Early stopping
        self.wait_early_stop = 0
        self.early_stop_patience = early_stop_patience

        # Reduce lr
        self.wait_reduce_lr = 0
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor

        if run_id:
            self._load_best()

    def _load_best(self):
        try:
            client = MlflowClient()
            metrics = client.get_metric_history(self.run_id, self.monitor)

            if not metrics:
                return

            # Sort by step in case they are not sorted
            metrics.sort(key=lambda m: m.step)

            # Find the minimum value of the monitor and its epoch
            losses = [(m.step, m.value) for m in metrics]
            self.best_epoch, self.best_value = min(losses, key=lambda x: x[1])
            last_epoch = metrics[-1].step

            # Count how many times the loss wasn't improving after self.reduce_lr_patience epochs
            count = 0
            streak = 0
            for i in range(1, len(losses)):
                if losses[i][1] >= losses[i-1][1]:
                    streak += 1
                else:
                    streak = 0

                if streak == self.reduce_lr_patience:
                    count += 1
                    streak = 0

            for _ in range(count):
                model.scheduler_lr.halve_learning_rate(reduce_factor=self.reduce_lr_factor)

            # how many epochs without improvement from the last learning rate change
            self.wait_reduce_lr = streak
            # how many epochs without improvement at all
            self.wait_early_stop = last_epoch - self.best_epoch

            print(f"[MLFlowEarlyStopping] Best val_loss: {self.best_value:.4f} at epoch {self.best_epoch}")
            print(f"[MLFlowEarlyStopping] Patience for LR reduction: {self.wait_reduce_lr}")
            print(f"[MLFlowEarlyStopping] Patience for early stopping: {self.wait_early_stop}")

        except Exception as e:
            print(f"[MLFlowEarlyStopping] Failed to load {self.monitor} history from MLflow: {e}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            return

        # Minimize the metric
        if monitor_value + self.min_delta < self.best_value:
            self.best_value = monitor_value
            self.best_epoch = epoch
            self.wait_early_stop = 0
            self.wait_reduce_lr = 0
        else:
            self.wait_early_stop += 1
            self.wait_reduce_lr += 1

        if self.wait_early_stop >= self.early_stop_patience:
            # Applies the early stop
            self.model.stop_training = True
            if self.verbose > 0:
                print(f"Epoch {epoch + 1}: early stopping (wait={self.wait_early_stop})")
                print(f'Best val_loss: {self.best_value:.4f} at epoch {self.best_epoch + 1}')

        if self.wait_reduce_lr >= self.reduce_lr_patience:
            # Applies the reduce lr cap
            new_lr_cap = model.scheduler_lr.halve_learning_rate(reduce_factor=self.reduce_lr_factor)
            self.wait_reduce_lr = 0
            if self.verbose > 0:
                print(f"Epoch {epoch + 1}: learning rate cap reduce to = {new_lr_cap}")

EarlyStopping_and_reduce_lr_cap = MLFlowEarlyStopping_and_reduce_lr_cap(run_id, monitor='val_loss',
                                                                early_stop_patience=6,
                                                                min_delta=0.0,
                                                                reduce_lr_factor=0.5,
                                                                reduce_lr_patience=3,
                                                                verbose=1)

# Define number of epochs to fit
epochs = 100

with run:
    # If new run then stores the initial params
    if not run_id:
        mlflow.log_param("initial_epoch", initial_epoch)
        mlflow.log_param("model_type", "Transformer")
        mlflow.log_param("target_maxlen", max_target_len)
        mlflow.log_param("input_dim_speech_feature_embedding",
                         (max_time_len, fft_length//2 + 1))
        mlflow.log_param("num_hid", num_hid)
        mlflow.log_param("num_feed_forward", num_feed_forward)
        mlflow.log_param("num_layers_enc", num_layers_enc)
        mlflow.log_param("num_layers_dec", num_layers_dec)
        mlflow.log_param("num_head", num_head)
        run_id = run.info.run_id

    # Train the model
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        # train_callback has to be after EarlyStopping_and_reduce_lr
        # (EarlyStopping_and_reduce_lr has to use loss only from previous epochs and
        # train_callback stores the loss from the current epoch)
        # validation_callback has to be before EarlyStopping_and_reduce_lr
        # (validation_callback stores de lr from the current epoch and then 
        # EarlyStopping_and_reduce_lr reduce the lr if needed)
        callbacks=[save_weights_and_optimizer_state(checkpoint_dir),
                   validation_callback,
                   EarlyStopping_and_reduce_lr_cap,
                   train_callback,
                   MetricsPlotCallback(run_id=run_id)],
        initial_epoch=initial_epoch
    )