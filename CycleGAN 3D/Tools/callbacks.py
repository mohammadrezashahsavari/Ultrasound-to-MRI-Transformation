from tensorflow.keras.callbacks import Callback

class SaveModel(Callback):
    def __init__(self, save_best_model_to, save_last_model_to):
        super(SaveModel, self).__init__()
        self.best_model_path = save_best_model_to
        self.last_model_path = save_last_model_to
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.model.save_weights(self.best_model_path, overwrite=True)
            print(f"Best model weights saved with validation loss: {val_loss:.4f}")
        self.model.save_weights(self.last_model_path, overwrite=True)
        # print(f"Last model weights saved with validation loss: {val_loss:.4f}")
