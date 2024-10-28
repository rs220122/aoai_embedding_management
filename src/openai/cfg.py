"""
Author: your name
Date: 2024-10-27 19:24:30
OpenAI エンベッディングモデルのYAMLファイルを読み込む
"""

import logging
import os
from enum import Enum

# third-party packages
import yaml

# user-defined packages
from pydantic import BaseModel

_MODEL_KEY = "models"
_ENDPOINT_KEY = "endpoint_environ"
_API_KEY_DICT_KEY = "api_key_environ"


class Resource(BaseModel):
    """
    OpenAIのエンドポイントとAPIキーのリスト
    """

    api_key: str
    endpoint: str

    class Config:
        allow_mutation = False

    @classmethod
    def create(cls, endpoint_environ: str, api_key_environ: str) -> "Resource":
        return cls(
            endpoint=os.environ[endpoint_environ], api_key=os.environ[api_key_environ]
        )

    def __repr__(self):
        return f"ENDPOINT={self.endpoint} api_key=[REDACTED]"

    def __str__(self):
        return f"ENDPOINT={self.endpoint} api_key=[REDACTED]"


class ModelQuotaData(BaseModel):
    """
    リソース情報とクォータとデプロイ名を管理する。
    """

    quota: int
    model_deployment_name: str
    resource: Resource

    class Config:
        allow_mutation = False

    @classmethod
    def create(
        cls, resource_info: dict[str, str], model_environ: str, quota: int
    ) -> "ModelQuotaData":
        resource = Resource.create(
            endpoint_environ=resource_info[_ENDPOINT_KEY],
            api_key_environ=resource_info[_API_KEY_DICT_KEY],
        )
        # quotaに入れる段階で、1K -> 1000に変換する。
        return cls(
            quota=quota * 1000,
            model_deployment_name=os.environ[model_environ],
            resource=resource,
        )


class ModelNameEnum(Enum):
    embedding_ada_002: str = "text-embedding-ada-002"
    embedding_3_small: str = "text-embedding-3-small"
    embedding_3_large: str = "text-embedding-3-large"

    @classmethod
    def values(cls) -> list[str]:
        """
        一覧を返す。
        """
        return [model.value for model in cls]


def load_max_token_dict() -> dict[ModelNameEnum, int]:
    return {
        ModelNameEnum.embedding_3_large: 8191,
        ModelNameEnum.embedding_3_small: 8191,
        ModelNameEnum.embedding_ada_002: 8191,
    }


def load_to_repo(cfg_path: str) -> dict[ModelNameEnum, dict[str, ModelQuotaData]]:

    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    model_names = cfg[_MODEL_KEY].keys()

    # モデル名がすべてModelNameと一致しているを確認する。
    warning_model_names = set(model_names) - set(ModelNameEnum.values())
    for m in warning_model_names:
        logging.warning(f"model name '{m}' in cfg is not defined in source code")

    not_define_model_names = set(ModelNameEnum.values()) - set(model_names)
    for m in not_define_model_names:
        logging.warning(f"model name '{m}' is not defined in config")

    model_dict: dict[ModelNameEnum, dict[str, ModelQuotaData]] = dict()
    for m in ModelNameEnum:
        resources_in_model = cfg[_MODEL_KEY][m.value]
        resource_dict: dict[str, ModelQuotaData] = {
            k: ModelQuotaData.create(**resources_in_model[k])
            for k in resources_in_model
        }
        model_dict[m] = resource_dict
    return model_dict


class Repo(object):
    # クラスインスタンス
    models: dict[ModelNameEnum, dict[str, ModelQuotaData]] = load_to_repo(
        os.environ["OPENAI_CFG_PATH"]
    )
    model_max_token_dict: dict[ModelNameEnum, int] = load_max_token_dict()

    @classmethod
    def get_model_identities(cls, model_name: ModelNameEnum) -> list[str]:

        return [k for k in cls.models[model_name]]

    @classmethod
    def get_model_data(
        cls, model_name: ModelNameEnum, model_identity: str
    ) -> ModelQuotaData:
        identities = cls.get_model_identities(model_name)
        if model_identity not in identities:
            raise ValueError(
                f"model identity '{model_identity}' in '{model_name}' is not implemented"
            )
        return cls.models[model_name][model_identity]

    @classmethod
    def get_models(cls) -> dict[ModelNameEnum, dict[str, ModelQuotaData]]:
        return cls.models

    @classmethod
    def get_model_quota(cls, model_name: ModelNameEnum, model_identity: str) -> int:
        return cls.get_model_data(model_name, model_identity).quota

    @classmethod
    def get_max_token(cls, model_name: ModelNameEnum) -> int:
        """
        そのモデルの最大のトークン数を計算する。
        これ以上のトークン数は一回でエンベッディングできないため、チャンクが必要
        """
        return cls.model_max_token_dict[model_name]
