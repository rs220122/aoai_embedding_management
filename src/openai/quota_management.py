"""
Author: your name
Date: 2024-10-27 22:53:11
"""

import math
import threading
from datetime import datetime, timedelta
from typing import Union

# third-party packages
from pydantic import BaseModel, Field

from src import exc

# user-defined packages
from . import cfg as openai_cfg


class ObtainedQuotaResponse(BaseModel):
    """
    クォータが確保できた場合は、このレスポンスを使う。
    """

    model_identity: str
    tokens: int
    model_name: openai_cfg.ModelNameEnum

    class Config:
        use_enum_values = True


class NoObtainedQuotaResponse(BaseModel):
    """
    クォータがない場合はこちらを返す。
    """

    model_name: openai_cfg.ModelNameEnum
    tokens: int
    wait_seconds: int = Field(
        ..., description="必要なクォータが解放されるまでの秒数を返す"
    )

    class Config:
        use_enum_values = True


class UsedQuota(BaseModel):
    """
    占有したクォータを記載
    """

    model_identity: str
    tokens: int
    timestamp: datetime


def calculate_wait_time(
    used_quotas: list[UsedQuota],
    request_tokens: int,
    now_available_quota: int,
    now_timestamp: datetime = None,
) -> int:
    """
    クォータ不足時に待ち時間を計算するメソッド。
    使用済みトークンデータを時間順に解放し、要求されたトークン数に達したらその待ち時間を返す。

    クォータ設定：

    全体のクォータ上限は「10K（キロトークン）」と仮定しています。
    現在すでに「7K」が使用済みであるため、「今利用可能なクォータ」は「3K」と設定しています。
    使用済みクォータの登録：

    used_quotasというリストに、過去に使用されたクォータ情報を追加します。
    追加した使用済みクォータは次のとおりです：
    1つ目のクォータ: 3Kトークン、使用開始時間は「2024年10月10日 00:00:50」
    2つ目のクォータ: 4Kトークン、使用開始時間は「2024年10月10日 00:01:05」
    このリストの中身は時間順に並べられ、最も早く解放されるものから順にトークンを計算していくことを前提としています。
    現在時刻の設定：

    now_timestampとして、現在の時刻を「2024年10月10日 00:01:30」に設定しています。この時刻において、リクエストに応じてトークンが確保できるか、待ち時間が必要かを判定します。
    リクエストトークンの設定：

    リクエストされたトークン数は「7K」トークンです。
    現在「3K」トークンが利用可能であるため、さらに「4K」トークンの解放が必要になります。
    calculate_wait_timeの動作確認
    この設定に基づき、calculate_wait_time関数は以下の順序で待ち時間を計算します：

    最初のクォータ確認：

    00:00:50に使用された「3K」トークンは、1分経過後の「00:01:50」に解放される見込みです。
    この時点で利用可能トークンは「3K + 3K = 6K」となり、リクエストトークン「7K」にはまだ不足しています。
    次のクォータ確認：

    次に、00:01:05に使用された「4K」トークンが、1分経過後の「00:02:05」に解放される見込みです。
    この時点でさらに「4K」トークンが追加され、合計「10K」トークンとなり、リクエストトークン数「7K」を満たします。
    待ち時間の計算：

    結果として、追加の「4K」トークンが解放される「00:02:05」までの待ち時間が求められます。
    now_timestampが「00:01:30」であるため、待ち時間は「35秒」となります。
    """
    if now_timestamp is None:
        now_timestamp = datetime.now()
    # 使用済みクォータを時間順にソート
    sorted_used_quotas = sorted(used_quotas, key=lambda x: x.timestamp)

    # 累計トークン数と最短待機時間の計算
    accumulated_tokens = now_available_quota
    for quota in sorted_used_quotas:
        accumulated_tokens += quota.tokens
        if accumulated_tokens >= request_tokens:
            wait_seconds = (
                quota.timestamp + timedelta(minutes=1) - now_timestamp
            ).total_seconds()
            return max(int(wait_seconds), 0)  # 負の待ち時間が出ないよう0と比較

    # 全クォータを解放しても足りない場合
    return math.inf


class OpenAIManagement:
    used_quotas: dict[openai_cfg.ModelNameEnum, dict[str, list[UsedQuota]]] = {
        model: {identity: [] for identity in identities}
        for model, identities in openai_cfg.Repo.get_models().items()
    }

    _quota_lock = threading.Lock()

    @classmethod
    def _remove_expired_quotas(
        cls, model_name: openai_cfg.ModelNameEnum, model_identity: str
    ):
        expiration_time = datetime.now() - timedelta(minutes=1)
        cls.used_quotas[model_name][model_identity] = [
            quota
            for quota in cls.used_quotas[model_name][model_identity]
            if quota.timestamp > expiration_time
        ]

    @classmethod
    def _obtain_quota(
        cls, model_name: openai_cfg.ModelNameEnum, model_identity: str, tokens: int
    ) -> Union[ObtainedQuotaResponse, NoObtainedQuotaResponse]:
        with cls._quota_lock:

            cls._remove_expired_quotas(model_name, model_identity)

            # 要求トークン数が全体のクォータ量を超えていたら、待機時間をinfとして返す。
            if openai_cfg.Repo.get_model_quota(model_name, model_identity) < tokens:
                return NoObtainedQuotaResponse(
                    model_name=model_name, tokens=tokens, wait_seconds=math.inf
                )

            # 利用可能なクォータを計算
            used_quota_total = sum(
                item.tokens for item in cls.used_quotas[model_name][model_identity]
            )
            available_quota = (
                openai_cfg.Repo.get_model_quota(model_name, model_identity)
                - used_quota_total
            )

            if available_quota >= tokens:
                # クォータ確保
                timestamp = datetime.now()
                cls.used_quotas[model_name][model_identity].append(
                    UsedQuota(
                        model_identity=model_identity,
                        tokens=tokens,
                        timestamp=timestamp,
                    )
                )
                return ObtainedQuotaResponse(
                    model_identity=model_identity, tokens=tokens, model_name=model_name
                )

            # クォータ不足時の待機時間
            wait_seconds = calculate_wait_time(
                cls.used_quotas[model_name][model_identity],
                request_tokens=tokens,
                now_available_quota=available_quota,
            )
            return NoObtainedQuotaResponse(
                model_name=model_name, tokens=tokens, wait_seconds=wait_seconds
            )

    @classmethod
    def get_quota(
        cls, model_name: openai_cfg.ModelNameEnum, tokens: int
    ) -> Union[ObtainedQuotaResponse, NoObtainedQuotaResponse]:
        model_identities = openai_cfg.Repo.get_model_identities(model_name=model_name)

        min_wait_seconds = math.inf
        res: NoObtainedQuotaResponse = None
        for model_identity in model_identities:
            response = cls._obtain_quota(model_name, model_identity, tokens)
            if isinstance(response, ObtainedQuotaResponse):
                return response  # クォータが確保できた場合
            if min_wait_seconds > response.wait_seconds:
                min_wait_seconds = response.wait_seconds
                res = response

        # すべての待ち時間がinfの場合は、そもそもデプロイ時のクォータ数が要求トークンに対して不足している。
        if min_wait_seconds == math.inf:
            # どのモデルでも要求トークン数が足りない場合
            raise exc.OpenAIQuotaAvailableExc(
                detail="要求トークン数{tokens}は、デプロイされているモデルのどのクォータよりも大きくなっています。デプロイモデルのクォータ数を引き上げてください。"
            )
        # 最小の待ち時間を返す。
        return res
