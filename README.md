# Python MLプロジェクト　テンプレート

## ファイル構成
| ディレクトリ   | 内容                                             |
|------------|------------------------------------------------|
| `installation` | 環境セットアップに使用するファイル (`Dockerfile`, `requirements.txt`, `conda_env.yaml` など) |
| `docs`         | ドキュメント (アーキテクチャ図、フローチャート、シーケンス図など)                  |
| `pipelines`    | 機械学習パイプラインの格納                                      |
| `src`          | ソースコードの格納                                          |
| `tests`        | テストコードの格納 (`pytest`を想定)                             |
| `notebooks`    | 実験用Jupyterノートブックの格納                                   |


## 環境セットアップ
例
```
pip install -r installation/requirements.txt
conda env create -f conda_env.yaml
```


## 実行方法
```
python main.py
```