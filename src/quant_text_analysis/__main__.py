"""quant_text_analysis パッケージの CLI エントリーポイント。"""

from __future__ import annotations

import importlib
import sys
from argparse import ArgumentParser
from typing import List


def _dispatch(module_path: str) -> int:
    """モジュールを読み込み `main` 関数を呼び出す。

    Args:
        module_path (str): `main` を含むサブコマンドモジュールのドット区切りパス。

    Returns:
        int: サブコマンドが返した終了コード。未指定の場合は 0。

    Raises:
        SystemExit: `main` が存在しない、または呼び出せない場合。
    """

    mod = importlib.import_module(module_path)
    entry = getattr(mod, "main", None)
    if entry is None or not callable(entry):
        raise SystemExit(f"Subcommand module '{module_path}' has no callable 'main'")
    ret = entry()  # 引数なしの main を呼び出す
    return int(ret) if isinstance(ret, int) else 0


def main(argv: List[str] | None = None) -> int:
    """CLI 引数を解釈して対応するサブコマンドを実行する。

    Args:
        argv (list[str] | None): サブコマンドとオプションの配列。None の場合は `sys.argv[1:]` を使用。

    Returns:
        int: サブコマンドの終了コード。
    """

    if argv is None:
        argv = sys.argv[1:]

    parser = ArgumentParser(prog="python -m quant_text_analysis", add_help=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sp_freq = subparsers.add_parser("freq", help="頻出語の集計と出力")
    sp_freq.set_defaults(_module="quant_text_analysis.commands.freq_cli")

    sp_phr = subparsers.add_parser("phrases", help="フレーズ候補の抽出")
    sp_phr.set_defaults(_module="quant_text_analysis.commands.phrases_cli")

    sp_clu = subparsers.add_parser("cluster", help="PPMI→SVD→球面k-means の一括実行")
    sp_clu.set_defaults(_module="quant_text_analysis.commands.cluster_cli")

    ns = parser.parse_args(argv)
    return _dispatch(getattr(ns, "_module"))


if __name__ == "__main__":  # pragma: no cover - CLI エントリーポイント
    sys.exit(main())
