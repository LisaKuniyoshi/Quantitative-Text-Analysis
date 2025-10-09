from __future__ import annotations
import hashlib
import json
import os
import pickle
from typing import Dict, Iterable, List, Optional, Tuple

from .data_types import TokenPolicy, DocResult, NLPBackend, Normalizer
from .perdoc import analyze_docs


def _policy_fingerprint(policy: TokenPolicy) -> dict[str, object]:
    # frozenset を JSON 化可能な list に落とす
    return {
        "target_pos": sorted(policy.target_pos),
        "exclude_ner": sorted(policy.exclude_ner),
        "exclude_propn": policy.exclude_propn,
        "exclude_aux": policy.exclude_aux,
        "keep_surface_for": sorted(policy.keep_surface_for),
        "alpha_regex": policy.alpha_regex,
        "forced_phrases": [list(p) for p in sorted(policy.forced_phrases)],
        "forced_joiner": policy.forced_joiner,
    }

def _make_key(texts: Iterable[str], policy: TokenPolicy, backend_id: str) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update((t or "").encode("utf-8"))
    h.update(json.dumps(_policy_fingerprint(policy), sort_keys=True, separators=(",", ":")).encode("utf-8"))
    h.update(backend_id.encode("utf-8"))
    return h.hexdigest()


def analyze_with_cache(
    backend: NLPBackend,
    normalizer: Normalizer,
    texts: List[str],
    policy: TokenPolicy,
    *,
    cache_dir: Optional[str] = None,
) -> Tuple[List[DocResult], List[Dict[str, float]]]:
    if cache_dir is None:
        return analyze_docs(backend, normalizer, texts,policy)
    os.makedirs(cache_dir, exist_ok=True)
    backend_id = getattr(backend, "model_name", backend.__class__.__name__)
    key = _make_key(texts, policy, str(backend_id))
    path = os.path.join(cache_dir, f"per_doc_freqs_{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    result = analyze_docs(backend, normalizer, texts, policy)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result
