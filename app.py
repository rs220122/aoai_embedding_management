"""
Author: your name
Date: 2024-10-27 23:07:27
"""

import math
import threading
from datetime import datetime, timedelta
from typing import Union

# third-party packages
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv()

# user-defined packages
from src import exc
from src.openai import cfg as openai_cfg
from src.openai.quota_management import OpenAIManagement, UsedQuota

app = FastAPI()


@app.get("/request_quota")
def request_quota(
    model_name: openai_cfg.ModelNameEnum = Query(description="要求するモデル名"),
    tokens: int = Query(
        description="要求するトークン数. 単位は、K(キロ)で指定。1000トークンを使いたい場合は、1を入れる。"
    ),
):
    if openai_cfg.Repo.get_max_token(model_name) < tokens:
        # 最大トークンバリデーション
        raise exc.OpenAIMaxTokenExc(
            f"{model_name} token limit is {openai_cfg.Repo.get_max_token(model_name)}"
        )
    response = OpenAIManagement.get_quota(model_name=model_name, tokens=tokens)

    return JSONResponse(
        content=response.dict(),
        status_code=200,
    )


@app.get("/used_quotas")
def get_used_quota(model_name: openai_cfg.ModelNameEnum):
    used_quotas = OpenAIManagement.used_quotas[model_name]

    response = dict()
    for model_identity, used_quota_per_model_identity in used_quotas.items():
        used_quota_per_model_identity: list[UsedQuota]
        response[model_identity] = sum(
            [u.tokens for u in used_quota_per_model_identity]
        )
    return JSONResponse(response, status_code=200)


@app.get("/model_quotas")
def get_model_quota(model_name: openai_cfg.ModelNameEnum):
    response = dict()
    for model_identity, model_quota_data in openai_cfg.Repo.get_models()[
        model_name
    ].items():
        response[model_identity] = model_quota_data.quota
    return JSONResponse(response, status_code=200)


@app.exception_handler(exc.BaseExc)
def response_error(req: Request, exc: exc.BaseExc):
    return JSONResponse(content=exc.to_json(), status_code=exc.status_code)
