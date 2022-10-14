import pytest, os
from deepClassifier.components import prepare_base_model
from deepClassifier.entity.config_entity import PrepareBaseModelConfig
from deepClassifier.components.prepare_base_model import PrepareBaseModel
from pathlib import Path
from deepClassifier.constants import *
from deepClassifier.utils import read_yaml, create_directories

class Test_Prepare_BaseModel:
    prepare_basemodel_config = PrepareBaseModelConfig(
        root_dir="artifacts/prepare_base_model",
        base_model_path="artifacts/prepare_base_model/base_model.h5",
        updated_base_model_path="artifacts/prepare_base_model/base_model_updated.h5",
        params_learning_rate=0.01,
        params_include_top=False,
        params_weights="imagenet",
        params_image_size=[224,224,3],
        params_classes=2
    )

    def test_base_model(self):
        base_model=PrepareBaseModel(config=self.prepare_basemodel_config)
        base_model.get_base_model()

        assert os.path.exists(self.prepare_basemodel_config.base_model_path)

class Test_Updated_Base_Model:
    prepare_basemodel_config = PrepareBaseModelConfig(
        root_dir="artifacts/prepare_base_model",
        base_model_path="artifacts/prepare_base_model/base_model.h5",
        updated_base_model_path="artifacts/prepare_base_model/base_model_updated.h5",
        params_learning_rate=0.01,
        params_include_top=False,
        params_weights="imagenet",
        params_image_size=[224,224,3],
        params_classes=2
    )

    def test_get_updated_base_model(self):
        base_model_updated=PrepareBaseModel(config=self.prepare_basemodel_config)
        base_model_updated.get_base_model()
        base_model_updated.get_updated_base_model()

        assert os.path.exists(self.prepare_basemodel_config.updated_base_model_path)