import pytest, os
from deepClassifier.entity.config_entity import PrepareCallbackConfig
from deepClassifier.components.prepare_callback import PrepareCallback
from pathlib import Path
from deepClassifier.constants import *
from deepClassifier.utils import read_yaml, create_directories

class Test_Prepare_Callback_Checkpoint:
    prepare_callback_config = PrepareCallbackConfig(
        root_dir="artifacts/prepare_callbacks",
        tensorboard_root_log_dir="artifacts/prepare_callbacks/tensorboard_log_dir",
        checkpoint_model_filepath="artifacts/prepare_callbacks/checkpoint_dir")

    def test_get_tb_ckpt_callbacks(self):
        prepare_callbacks=PrepareCallback(config=self.prepare_callback_config)
        prepare_callbacks.get_tb_ckpt_callbacks()

        assert os.path.exists(self.prepare_callback_config.checkpoint_model_filepath)

class Test_Prepare_Callback_Tensorboard_Log_Dir:
    prepare_callback_config = PrepareCallbackConfig(
        root_dir="artifacts/prepare_callbacks",
        tensorboard_root_log_dir="artifacts/prepare_callbacks/tensorboard_log_dir",
        checkpoint_model_filepath="artifacts/prepare_callbacks/checkpoint_dir")

    def test_get_tb_log_dir(self):
        prepare_callbacks=PrepareCallback(config=self.prepare_callback_config)
        prepare_callbacks.get_tb_ckpt_callbacks()

        assert os.path.exists(self.prepare_callback_config.tensorboard_root_log_dir)