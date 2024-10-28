"""
Author: your name
Date: 2024-10-28 13:41:53
例外クラスの定義を行う
"""

from http import HTTPStatus
from typing import Union

# third-party packages


# user-defined packages


class BaseExc(Exception):
    code: int = 500
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST

    def __init__(self, detail: str):
        self.detail = detail

    def to_json(self) -> dict[str, Union[int, str]]:
        return {"code": self.__class__.code, "detail": self.detail}


class OpenAIMaxTokenExc(BaseExc):
    code: int = 100
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST


class OpenAIQuotaAvailableExc(BaseExc):
    code: int = 101
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST


if __name__ == "__main__":
    pass
