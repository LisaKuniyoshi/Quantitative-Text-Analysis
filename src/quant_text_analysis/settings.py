# quant_text_analysis/settings.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from .config import default_columns, default_token_policy
from .data_types import Columns, TokenPolicy

# 既定パス（プロジェクトのレイアウトに依存）
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
CACHE_DIR: Path = DATA_DIR / "cache"
OUT_DIR: Path = PROJECT_ROOT / "outputs"
CSV_PATH: Path = RAW_DIR / "エクスポートされたアイテム.csv"

@dataclass(frozen=True)
class Settings:
    # パス
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    raw_dir: Path = RAW_DIR
    cache_dir: Path = CACHE_DIR
    out_dir: Path = OUT_DIR
    csv_path: Path = CSV_PATH

    # テキスト前処理
    spacy_model: str = "en_core_web_sm"

    # 語彙選定（PPMI 前段）
    top_n: int = 10_000
    min_docs: int = 7

    # 埋め込み次元（非対称PPMI→SVD）
    svd_dim: int = 200

    # クラスタリング
    k_list: Tuple[int, ...] = (12, 16, 20)
    n_init: int = 20
    max_iter: int = 300
    random_seed: int = 42

    # 出力
    top_words_per_cluster: int = 20

    # 列名・正規化ポリシ（注入ポイント）
    @property
    def columns(self) -> Columns:
        return default_columns()

    @property
    def token_policy(self) -> TokenPolicy:
        return default_token_policy()

    def ensure_out_dir(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self.out_dir
