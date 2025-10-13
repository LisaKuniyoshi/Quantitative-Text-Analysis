# quant_text_analysis/cluster/io.py
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

def save_vocab(out_dir: Path, vocab: List[str]) -> None:
    """語彙リストを JSON 形式で保存する。

    Parameters
    ----------
    out_dir : Path
        出力先ディレクトリ。
    vocab : List[str]
        語彙リスト。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(
        json.dumps({"vocab": vocab}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def save_ppmi(out_dir: Path, ppmi_wd: sp.spmatrix, ppmi_ww: sp.spmatrix) -> None:
    """PPMI 行列を NPZ 形式で保存する。

    Parameters
    ----------
    out_dir : Path
        出力先ディレクトリ。
    ppmi_wd : scipy.sparse.spmatrix
        語×文書の PPMI 行列。
    ppmi_ww : scipy.sparse.spmatrix
        語×語の PPMI 行列。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "PPMI_word_doc_VxD.npz", sp.csr_matrix(ppmi_wd))
    sp.save_npz(out_dir / "PPMI_word_word_VxV.npz", sp.csr_matrix(ppmi_ww))

def save_top_terms(
    out_dir: Path,
    k: int,
    top: Dict[int, List[Tuple[str, float]]]
) -> Path:
    """クラスタ上位語リストを CSV 形式で保存する。

    Parameters
    ----------
    out_dir : Path
        出力先ディレクトリ。
    k : int
        クラスタ数。
    top : Dict[int, List[Tuple[str, float]]]
        クラスタ ID ごとの語と類似度のリスト。

    Returns
    -------
    Path
        生成された CSV ファイルのパス。
    """
    rows: List[Tuple[int, int, str, float]] = []
    for c, pairs in top.items():
        for rank, (w, s) in enumerate(pairs, start=1):
            rows.append((c, rank, w, s))
    df = pd.DataFrame(rows, columns=["cluster", "rank", "word", "cos_to_centroid"])
    path = out_dir / f"top_terms_k{k}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path

def save_labels(out_dir: Path, k: int, vocab: List[str], labels: np.ndarray) -> Path:
    """語彙とクラスタ割当を CSV 形式で保存する。

    Parameters
    ----------
    out_dir : Path
        出力先ディレクトリ。
    k : int
        クラスタ数。
    vocab : List[str]
        語彙リスト。
    labels : numpy.ndarray
        クラスタラベル配列。

    Returns
    -------
    Path
        生成された CSV ファイルのパス。
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
    silhouette: Optional[float],
    stability_jaccard: Optional[float]
) -> Path:
    """クラスタ評価指標を JSON 形式で保存する。

    Parameters
    ----------
    out_dir : Path
        出力先ディレクトリ。
    k : int
        クラスタ数。
    inertia : float
        cos 慣性。
    silhouette : Optional[float]
        cos シルエット。NaN の場合は None に変換。
    stability_jaccard : Optional[float]
        トップ語集合の Jaccard 安定性。

    Returns
    -------
    Path
        生成された JSON ファイルのパス。
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
        "stability_top_terms_jaccard": _nan_to_none(stability_jaccard),
    }
    path = out_dir / f"metrics_k{k}.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def save_cluster_ratio(out_dir: Path, k: int, M: np.ndarray) -> Path:
    """文書×クラスタ比率行列を NPY 形式で保存する。

    Parameters
    ----------
    out_dir : Path
        出力先ディレクトリ。
    k : int
        クラスタ数。
    M : numpy.ndarray
        文書×クラスタ比率行列。

    Returns
    -------
    Path
        生成された NPY ファイルのパス。
    """
    path = out_dir / f"abstract_ratio_k{k}.npy"
    np.save(path, M)
    return path
