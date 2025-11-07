"""Grouping helpers used by frequency and clustering analyses."""

from __future__ import annotations

def period_group_year(y: int) -> list[str]:
    """発行年から集計用の期間ラベルを生成する。

    Args:
        y (int): 発行年。

    Returns:
        str: 期間ラベル。該当しない場合は例外を送出する。
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
        list[str] : 推定されたカテゴリのリスト。
    """
    out: list[str] = []
    for raw in tags.split(";"):
        token = raw.strip()
        if token in _METHOD_TAGS:
            out.append(token)

    return out