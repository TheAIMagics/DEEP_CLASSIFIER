import pytest, os
from deepClassifier.components import prepare_base_model
from deepClassifier.entity.config_entity import PrepareBaseModelConfig
from deepClassifier.components.prepare_base_model import PrepareBaseModel
from pathlib import Path
from deepClassifier.constants import *
from deepClassifier.utils import read_yaml, create_directories

class Test_Prepare_BaseModel:
    def test_base_model(self):
        config_filepath = CONFIG_FILE_PATH
        params_filepath = PARAMS_FILE_PATH
        cfp = read_yaml(config_filepath)
        pfp = read_yaml(params_filepath)

        prepare_basemodel_config = PrepareBaseModelConfig(
            root_dir=cfp.prepare_base_model.root_dir,
            base_model_path=cfp.prepare_base_model.base_model_path,
            updated_base_model_path=cfp.prepare_base_model.updated_base_model_path,
            params_learning_rate=pfp.LEARNING_RATE,
            params_include_top=pfp.INCLUDE_TOP,
            params_weights=pfp.WEIGHTS,
            params_image_size=pfp.IMAGE_SIZE,
            params_classes=pfp.CLASSES
        )

        base_model=PrepareBaseModel(config=prepare_basemodel_config)
        base_model.get_base_model()

        assert os.path.exists(prepare_basemodel_config.base_model_path)

class Test_Updated_Base_Model:
    def test_get_updated_base_model(self):
        config_filepath = CONFIG_FILE_PATH
        params_filepath = PARAMS_FILE_PATH
        cfp = read_yaml(config_filepath)
        pfp = read_yaml(params_filepath)

        prepare_basemodel_config = PrepareBaseModelConfig(
            root_dir=cfp.prepare_base_model.root_dir,
            base_model_path=cfp.prepare_base_model.base_model_path,
            updated_base_model_path=cfp.prepare_base_model.updated_base_model_path,
            params_learning_rate=pfp.LEARNING_RATE,
            params_include_top=pfp.INCLUDE_TOP,
            params_weights=pfp.WEIGHTS,
            params_image_size=pfp.IMAGE_SIZE,
            params_classes=pfp.CLASSES
        )

        base_model_updated=PrepareBaseModel(config=prepare_basemodel_config)
        base_model_updated.get_base_model()
        base_model_updated.get_updated_base_model()

        assert os.path.exists(prepare_basemodel_config.updated_base_model_path)