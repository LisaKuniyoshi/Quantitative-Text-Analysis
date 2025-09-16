from __future__ import annotations
from typing import Optional
import pandas as pd

def period_group_year(y: Optional[int]) -> Optional[str]:
    if y is None or pd.isna(y):
        return None
    yi = int(y)
    if 2014 <= yi <= 2021:
        return "2014–2021"
    if 2022 <= yi <= 2023:
        return "2022–2023"
    if 2024 <= yi <= 2025:
        return "2024–2025"
    return None

def method_group(tags: Optional[str]) -> Optional[str]:
    if tags is None:
        return None
    s = tags.lower()
    if "qual" in s:
        return "qual"
    if "quan" in s:
        return "quan"
    if "theoretic" in s:
        return "theoretic"
    if "review" in s:
        return "review"
    return None