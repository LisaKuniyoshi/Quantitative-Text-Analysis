# quant_text_analysis/cluster/io.py
from __future__ import annotations
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

def save_vocab(out_dir: Path, vocab: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(
        json.dumps({"vocab": vocab}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def save_ppmi(out_dir: Path, ppmi_wd: sp.spmatrix, ppmi_ww: sp.spmatrix) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out_dir / "PPMI_word_doc_VxD.npz", sp.csr_matrix(ppmi_wd))
    sp.save_npz(out_dir / "PPMI_word_word_VxV.npz", sp.csr_matrix(ppmi_ww))

def save_top_terms(
    out_dir: Path,
    k: int,
    top: Dict[int, List[Tuple[str, float]]]
) -> Path:
    rows: List[Tuple[int, int, str, float]] = []
    for c, pairs in top.items():
        for rank, (w, s) in enumerate(pairs, start=1):
            rows.append((c, rank, w, s))
    df = pd.DataFrame(rows, columns=["cluster", "rank", "word", "cos_to_centroid"])
    path = out_dir / f"top_terms_k{k}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    return path

def save_labels(out_dir: Path, k: int, vocab: List[str], labels: np.ndarray) -> Path:
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
    path = out_dir / f"abstract_ratio_k{k}.npy"
    np.save(path, M)
    return path
