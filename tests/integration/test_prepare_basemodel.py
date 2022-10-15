import pytest, os
from deepClassifier.components import prepare_base_model
from deepClassifier.entity.config_entity import PrepareBaseModelConfig
from deepClassifier.components.prepare_base_model import PrepareBaseModel
from pathlib import Path
from deepClassifier.constants import *
from deepClassifier.utils import read_yaml

class Test_Prepare_BaseModel:
    config_fp = read_yaml(CONFIG_FILE_PATH)
    params_fp = read_yaml(PARAMS_FILE_PATH)

    prepare_basemodel_config = PrepareBaseModelConfig(
            root_dir=config_fp.prepare_base_model.root_dir,
            base_model_path=config_fp.prepare_base_model.base_model_path,
            updated_base_model_path=config_fp.prepare_base_model.updated_base_model_path,
            params_learning_rate=params_fp.LEARNING_RATE,
            params_include_top=params_fp.INCLUDE_TOP,
            params_weights=params_fp.WEIGHTS,
            params_image_size=params_fp.IMAGE_SIZE,
            params_classes=params_fp.CLASSES
        )

    def test_base_model(self):
        base_model=PrepareBaseModel(config=self.prepare_basemodel_config)
        base_model.get_base_model()

        assert os.path.exists(self.prepare_basemodel_config.base_model_path)

    def test_get_updated_base_model(self):
        base_model_updated=PrepareBaseModel(config=self.prepare_basemodel_config)
        base_model_updated.get_base_model()
        base_model_updated.get_updated_base_model()

        assert os.path.exists(self.prepare_basemodel_config.updated_base_model_path)