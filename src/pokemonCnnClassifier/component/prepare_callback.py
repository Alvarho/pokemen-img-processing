import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pokemonCnnClassifier.entity.config_entity import PrepareCallbackConfig


class PrepareCallback:
    def __init__(self, config: PrepareCallbackConfig):
        self.config = config


    
    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    

    @property
    def _create_ckpt_callbacks(self):
        return tf.train.latest_checkpoint(self.config.checkpoint_model_filepath, latest_filename=None)

    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]
