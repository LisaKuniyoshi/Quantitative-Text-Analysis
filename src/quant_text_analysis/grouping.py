"""Grouping helpers used by frequency and clustering analyses."""

from __future__ import annotations

from typing import Iterable, Sequence


def period_group_year(y: int) -> list[str]:
    """発行年から集計用の期間ラベルを生成する。

    Args:
        y (int): 発行年。

    Returns:
        list[str]: 該当する期間ラベルを 1 要素のリストで返す。

    Raises:
        ValueError: 期待される期間の範囲外の年が渡された場合。
    """
    if 2014 <= y <= 2021:
        return ["2014–2021"]
    if 2022 <= y <= 2023:
        return ["2022–2023"]
    if 2024 <= y <= 2025:
        return ["2024–2025"]

    raise ValueError("Year out of expected range: {}".format(y))


METHOD_CODE_TO_LABEL: dict[str, str] = {
    "qual": "質",
    "quan": "量",
    "review": "レ",
    "theoretic": "理",
    "other": "その他",
}

_METHOD_TAGS = frozenset(METHOD_CODE_TO_LABEL.keys())
_DEFAULT_METHOD_LABEL = METHOD_CODE_TO_LABEL["other"]


def method_group(tags: str | None) -> list[str]:
    """手動タグから研究手法カテゴリを推定する。"""

    if not isinstance(tags, str):
        return []

    out: list[str] = []
    for raw in tags.split(";"):
        token = raw.strip()
        if token in _METHOD_TAGS:
            out.append(token)

    return out


def method_labels_for_display(method_codes: Sequence[str]) -> list[str]:
    """Convert method code tokens into human-friendly labels."""

    labels: list[str] = []
    seen: set[str] = set()
    for code in method_codes:
        label = METHOD_CODE_TO_LABEL.get(code, code)
        if label in seen:
            continue
        labels.append(label)
        seen.add(label)
    return labels


def method_label_string(method_codes: Sequence[str] | None) -> str:
    """Return a single string label for display, defaulting to その他 when empty."""

    if not method_codes:
        return _DEFAULT_METHOD_LABEL

    labels = method_labels_for_display(method_codes)
    if not labels:
        return _DEFAULT_METHOD_LABEL
    return "・".join(labels)


def method_codes_from_labels(labels: Iterable[str]) -> list[str]:
    """Best-effort reverse lookup from labels back to method codes."""

    inverse = {label: code for code, label in METHOD_CODE_TO_LABEL.items()}
    out: list[str] = []
    for label in labels:
        code = inverse.get(label, label)
        if code in _METHOD_TAGS:
            out.append(code)
    return out
