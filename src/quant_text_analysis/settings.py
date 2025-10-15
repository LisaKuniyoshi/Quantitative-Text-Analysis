# quant_text_analysis/settings.py
from __future__ import annotations

from dataclasses import dataclass
import datetime
from pathlib import Path
from typing import Tuple

from .config import default_columns, default_token_policy
from .data_types import Columns, TokenPolicy

# 既定パス（プロジェクトのレイアウトに依存）
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
CACHE_DIR: Path = DATA_DIR / "cache"
OUT_DIR: Path = PROJECT_ROOT / "outputs" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH: Path = RAW_DIR / "エクスポートされたアイテム.csv"

@dataclass(frozen=True)
class Settings:
    """プロジェクト全体の設定値を保持する不変データクラス。

    Attributes:
        project_root (Path): プロジェクトルートディレクトリ。
        data_dir (Path): データフォルダへのパス。
        raw_dir (Path): 生データ格納先のパス。
        cache_dir (Path): 中間生成物を保存するキャッシュディレクトリ。
        out_dir (Path): 出力ディレクトリのルート。
        csv_path (Path): 入力 CSV ファイルのパス。
        spacy_model (str): 利用する spaCy モデル名。
        top_n (int): 語彙選定時に保持する上位語数。
        min_docs (int): 語彙選定時の最小文書出現数。
        svd_dim (int): SVD における埋め込み次元。
        k_list (Tuple[int, ...]): 評価するクラスタ数の候補。
        n_init (int): k-means の初期化回数。
        max_iter (int): k-means の最大反復回数。
        random_seed (int): 乱数シード値。
        top_words_per_cluster (int): クラスタごとの上位語数。
    """

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
        """使用する列名設定を返す。

        Returns
        -------
        Columns
            使用する列名設定を表す `Columns` インスタンス。
        """
        return default_columns()

    @property
    def token_policy(self) -> TokenPolicy:
        """トークン正規化の設定を返す。

        Returns
        -------
        TokenPolicy
            トークン正規化に用いる `TokenPolicy` 設定。
        """
        return default_token_policy()

    def ensure_out_dir(self) -> Path:
        """出力ディレクトリを確実に作成して返す。

        Returns
        -------
        Path
            確実に存在する出力ディレクトリのパス。
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self.out_dir
