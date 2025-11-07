"""Grouping helpers used by frequency and clustering analyses."""

from __future__ import annotations

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


_METHOD_TAGS = {"qual", "quan", "review", "theoretic", "other"}


def method_group(tags: str) -> list[str]:
    """手動タグから研究手法カテゴリを推定する。

    Args:
        tags (str): セミコロン区切りの手法タグ文字列。

    Returns:
        list[str]: 認識された研究手法カテゴリのリスト。
            `_METHOD_TAGS` に含まれるトークンのみが保持され、同一文書が
            複数のカテゴリに属する場合は複数要素を返す。
    """
    out: list[str] = []
    for raw in tags.split(";"):
        token = raw.strip()
        if token in _METHOD_TAGS:
            out.append(token)

    return out