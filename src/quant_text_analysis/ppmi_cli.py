# quant_text_analysis/ppmi_cli.py
from __future__ import annotations

from pathlib import Path
from typing import List

from .config import default_columns, default_token_policy
from .io_loader import load_df
from .nlp_backend import SpacyBackend
from .normalize import build_normalizer
from .ppmi import compute_ppmi_with_cache

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
CACHE_DIR: Path = DATA_DIR / "cache"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

CSV_PATH: Path = RAW_DIR / "エクスポートされたアイテム.csv"
OUT_DIR: Path = OUTPUTS_DIR / "ppmi"

TOP_N: int = 10_000
MIN_DOCS: int = 7
SPACY_MODEL: str = "en_core_web_sm"

def main() -> None:
    cols = default_columns()
    policy = default_token_policy()

    df = load_df(str(CSV_PATH), cols)
    abstracts: List[str] = df["abstract"].fillna("").astype(str).tolist()

    backend = SpacyBackend(model=SPACY_MODEL)
    normalizer = build_normalizer(policy)

    out = compute_ppmi_with_cache(
        backend, normalizer, abstracts, policy,
        cache_dir=str(CACHE_DIR),
        top_n=TOP_N, min_docs=MIN_DOCS,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import json, scipy.sparse as sp
    json.dump({"vocab": out.vocab}, open(OUT_DIR / "vocab.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    sp.save_npz(OUT_DIR / "PPMI_word_doc_VxD.npz", out.ppmi_word_doc)
    sp.save_npz(OUT_DIR / "PPMI_word_word_VxV.npz", out.ppmi_word_word)

    print("PPMI done.")
    print(f"- vocab size: {len(out.vocab)}")
    print(f"- docs      : {len(out.doc_ids)}")
    print(f"- outputs   : {OUT_DIR}")

if __name__ == "__main__":
    main()
