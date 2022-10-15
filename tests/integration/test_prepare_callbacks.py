import pytest, os
from deepClassifier.entity.config_entity import PrepareCallbackConfig
from deepClassifier.components.prepare_callback import PrepareCallback
from pathlib import Path
from deepClassifier.constants import *
from deepClassifier.utils import read_yaml, create_directories

class Test_Prepare_Callback:
    config_fp = read_yaml(CONFIG_FILE_PATH)
    prepare_callback_config = PrepareCallbackConfig(
            root_dir=config_fp.prepare_callbacks.root_dir,
            tensorboard_root_log_dir=config_fp.prepare_callbacks.tensorboard_root_log_dir,
            checkpoint_model_filepath=config_fp.prepare_callbacks.checkpoint_model_filepath)

    def test_get_tb_ckpt_callbacks(self):
        prepare_callbacks=PrepareCallback(config=self.prepare_callback_config)
        prepare_callbacks.get_tb_ckpt_callbacks()

        assert os.path.exists(self.prepare_callback_config.tensorboard_root_log_dir)

    def test_get_tb_ckpt_callbacks(self):
        prepare_callbacks=PrepareCallback(config=self.prepare_callback_config)
        prepare_callbacks.get_tb_ckpt_callbacks()

        checkpoint_path = self.prepare_callback_config.checkpoint_model_filepath
        head,tail = os.path.split(checkpoint_path)
        
        assert os.path.exists(head)
