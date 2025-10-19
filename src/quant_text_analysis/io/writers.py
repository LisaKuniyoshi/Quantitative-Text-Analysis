# quant_text_analysis/cluster/io.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

def save_vocab(out_dir: Path, vocab: List[str]) -> None:
    """語彙リストを JSON 形式で保存する。

    Args:
        out_dir (Path): 出力先ディレクトリ。
        vocab (list[str]): 語彙リスト。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(
        json.dumps({"vocab": vocab}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def save_cluster_terms(
    out_dir: Path,
    k: int,
    vocab: List[str],
    labels: np.ndarray,
    silhouette_scores: np.ndarray,
) -> Path:
    """クラスタ語リストをシルエット指標付きで CSV に保存する。

    Args:
        out_dir (Path): 出力先ディレクトリ。
        k (int): クラスタ数。
        vocab (list[str]): 語彙リスト。
        labels (numpy.ndarray): 語彙に対応するクラスタラベル。
        silhouette_scores (numpy.ndarray): 各語のシルエット値。

    Returns:
        Path: 生成された CSV ファイルのパス。
    """
    df = pd.DataFrame(
        {
            "cluster": labels.astype(int),
            "word": vocab,
            "silhouette_cos": silhouette_scores.astype(float),
        }
    )
    df.sort_values(["cluster", "silhouette_cos"], ascending=[True, False], inplace=True)
    path = out_dir / f"cluster_terms_k{k}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path

def save_labels(out_dir: Path, k: int, vocab: List[str], labels: np.ndarray) -> Path:
    """語彙とクラスタ割当を CSV 形式で保存する。

    Args:
        out_dir (Path): 出力先ディレクトリ。
        k (int): クラスタ数。
        vocab (list[str]): 語彙リスト。
        labels (numpy.ndarray): クラスタラベル配列。

    Returns:
        Path: 生成された CSV ファイルのパス。
    """
    df = pd.DataFrame({"word": vocab, "cluster": labels.astype(int)})
    path = out_dir / f"labels_k{k}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path

def save_metrics(
    out_dir: Path,
    k: int,
    *,
    inertia: float,
    silhouette: Optional[float]
) -> Path:
    """クラスタ評価指標を JSON 形式で保存する。

    Args:
        out_dir (Path): 出力先ディレクトリ。
        k (int): クラスタ数。
        inertia (float): cos 慣性。
        silhouette (float | None): cos シルエット。NaN は None に変換。

    Returns:
        Path: 生成された JSON ファイルのパス。
    """
    def _nan_to_none(x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        # x は float を想定（NaN も float）。NaN のときのみ None に落とす
        return None if isinstance(x, float) and math.isnan(x) else float(x)

    rec = {
        "k": int(k),
        "inertia_cos": float(inertia),
        "silhouette_cos": _nan_to_none(silhouette),
    }
    path = out_dir / f"metrics_k{k}.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def save_cluster_ratio(out_dir: Path, k: int, M: np.ndarray) -> Path:
    """文書×クラスタ比率行列を NPY 形式で保存する。

    Args:
        out_dir (Path): 出力先ディレクトリ。
        k (int): クラスタ数。
        M (numpy.ndarray): 文書×クラスタ比率行列。

    Returns:
        Path: 生成された NPY ファイルのパス。
    """
    path = out_dir / f"abstract_ratio_k{k}.npy"
    np.save(path, M)
    return path
