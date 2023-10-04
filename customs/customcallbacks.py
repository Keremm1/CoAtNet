from keras.callbacks import Callback

class ValACCEarlyStopping(Callback):
    def __init__(self, monitor='val_accuracy', baseline_metric='accuracy', patience=3):
        super().__init__()
        self.monitor = monitor
        self.baseline_metric = baseline_metric
        self.patience = patience
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_val_metric = logs.get(self.monitor)
        current_baseline_metric = logs.get(self.baseline_metric)
        if current_val_metric is None or current_baseline_metric is None:
            raise ValueError(f"Cannot find metrics '{self.monitor}' or '{self.baseline_metric}' in logs.")

        if current_val_metric < current_baseline_metric:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(f"\nEarly stopping: Training stopped at epoch {epoch} because {self.monitor} is less than {self.baseline_metric}.")

        else:
            self.wait = 0

