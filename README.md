# Quantitative Text Analysis

論文要旨などの英語テキストコーパスに対して、定量的なテキスト分析を行うPythonパッケージです。

## 概要

本プロジェクトは、以下の主要な分析機能を提供します：

1. **頻出語分析** (`freq`) - 文書内相対頻度に基づく頻出語のランキング
2. **フレーズ抽出** (`phrases`) - Gensim Phrasesを用いたbigram/trigramの候補抽出
3. **単語クラスタリング** (`cluster`) - 文書×語頻度を SVD で次元削減し、球面 k-means で単語をクラスタリング
4. **コード×手法クロス集計** (`cross_table`) - 文書単位でコード出現有無を集計し、研究手法カテゴリと対比
5. **多項ロジット推定と周辺効果比較** (`mnlr`) - コード出現データを多項ロジスティック回帰で推定し、年×手法ごとの効果とペアワイズ差を評価

## 必要な環境

- Python 3.12.10
- 主要な依存パッケージ：
  - pandas ~=2.3
  - spacy ~=3.7
  - gensim ~=4.3
  - scikit-learn ~=1.5
  - breame ~=0.1.2
  - statsmodels ~=0.14
  - matplotlib ~=3.10

## インストール

1. リポジトリをクローンします：
```bash
git clone https://github.com/Slaine00/Quantitative-Text-Analysis.git
cd Quantitative-Text-Analysis
```

2. 依存パッケージをインストールします：
```bash
pip install -r requirements.txt
```

3. パッケージをインストールします：
```bash
pip install -e .
```

## データの準備

分析対象のデータは、以下のパスに配置してください：
- デフォルトパス: `data/raw/エクスポートされたアイテム.csv`

CSVファイルには以下の列が必要です：
- `abstract`: 論文の要旨（英語テキスト）
- `year`: 出版年
- `manual_tags`: 手作業でのタグ付け（研究手法の分類など）

## 使用方法

### 1. 頻出語分析 (freq)

全体、年代別、研究手法別の頻出語ランキングを算出します。

```bash
python -m quant_text_analysis freq
```

**出力ファイル:**
- `outputs/{タイムスタンプ}/top_words_overall.csv` - 全体の頻出語
- `outputs/{タイムスタンプ}/top_words_period_{グループ名}.csv` - 年代別の頻出語
- `outputs/{タイムスタンプ}/top_words_method_{グループ名}.csv` - 研究手法別の頻出語

**年代グループ:**
- 2014–2021
- 2022–2023
- 2024–2025

**研究手法グループ:**
- qual (質的研究)
- quan (量的研究)
- theoretic (理論研究)
- review (レビュー研究)
- other (その他)

### 2. フレーズ抽出 (phrases)

Gensim Phrasesを使用して、bigramとtrigramの候補を抽出します。

```bash
python -m quant_text_analysis phrases
```

**出力ファイル:**
- `outputs/{タイムスタンプ}/phrases_gensim.csv` - 抽出されたフレーズ候補とスコア

**特徴:**
- 英米表記の統一（breameライブラリ使用）
- 接続語を考慮したフレーズ学習
- モデルスコアと実コーパスでの使用統計を結合

### 3. 単語クラスタリング (cluster)

語×文書頻度行列から Truncated SVD で語埋め込みを生成し、L2 正規化後に球面 k-means でクラスタリングを実行します。

```bash
python -m quant_text_analysis cluster
```

**出力ファイル:**
- `outputs/{タイムスタンプ}/vocab.json` - 語彙リスト
- `outputs/{タイムスタンプ}/metrics.csv` - 各次元・クラスタ設定のサマリ
- `outputs/{タイムスタンプ}/svd_dim_{次元}/cluster_terms_k{K}.csv` - クラスタごとの語とシルエット値
- `outputs/{タイムスタンプ}/svd_dim_{次元}/labels_k{K}.csv` - 単語のクラスタ割当
- `outputs/{タイムスタンプ}/svd_dim_{次元}/metrics_k{K}.json` - クラスタリング評価指標
- `outputs/{タイムスタンプ}/svd_dim_{次元}/abstract_ratio_k{K}.npy` - 文書×クラスタ比率行列

**デフォルト設定:**
- クラスタ数: k=16, 19, 21, 25, 28, 31, 34, 37
- SVD次元数: 25（`svd_dim_list` で複数指定可）
- k-meansの初期化回数: 20
- 最大反復回数: 300

### 4. コード×手法クロス集計 (cross_table)

文書内にコードが一度でも出現したかを基準に、コードと研究手法のクロス集計を生成します。

```bash
python -m quant_text_analysis cross_table
```

**出力ファイル:**
- `outputs/{タイムスタンプ}/code_method_crosstab_docs.csv` - コード×研究手法（文書数）のクロス表

**特徴:**
- `config.CODE_MAP` に定義したコード集合を利用
- トークン化は既存のキャッシュを再利用しつつ再計算
- 手法ラベルは `grouping.method_group` の分類を適用

### 5. 多項ロジット推定と可視化 (mnlr)

文書トークンをコードに展開し、多項ロジット（MNLogit）で年・研究手法の効果を推定します。

```bash
python -m quant_text_analysis mnlr
```

**出力ファイル:**
- `outputs/{タイムスタンプ}/mnlogit_summary.txt` - MNLogit 推定結果の全文サマリ
- `outputs/{タイムスタンプ}/margeff_year_centered.csv` - `year_centered` の平均限界効果と標準誤差
- `outputs/{タイムスタンプ}/pairwise_ame_mnlogit.csv` - 手法カテゴリ間の平均限界効果差に対するペアワイズ検定結果

**特徴:**
- 文書 ID をクラスタとするロバスト共分散推定を実施
- `pairwise_ame_mnlogit` を用いて手法カテゴリ間の平均限界効果差を多重比較補正付きで評価
- `t_test_pairwise_mnlogit` により係数レベルでのペアワイズ t 検定も実行可能（スクリプトから呼び出し）
- コード定義は `config.CODE_MAP` を参照

ペアワイズ検定ユーティリティ（`quant_text_analysis.mnlr.statsmodels_fork`）は、statsmodels の公開 API のみを利用しており、平均限界効果または係数差についてロバスト共分散を尊重した検定結果を DataFrame で取得できます。*** End Patch

## 設定のカスタマイズ

設定は `src/quant_text_analysis/settings.py` の `Settings` クラスで管理されています。

主要な設定項目：

```python
# 入力データ
csv_path: Path = RAW_DIR / "エクスポートされたアイテム.csv"

# spaCyモデル
spacy_model: str = "en_core_web_sm"

# 語彙選定
top_n: int = 10_000          # 保持する上位語数
min_docs: int = 4            # 最小文書出現数

# 埋め込み
svd_dim_list: Tuple[int, ...] = (25,)  # 試行する SVD 次元

# クラスタリング
k_list: Tuple[int, ...] = (16, 19, 21, 25, 28, 31, 34, 37)  # クラスタ数候補
n_init: int = 20             # k-means初期化回数
max_iter: int = 300          # 最大反復回数
random_seed: int = 42        # 乱数シード
```

## プロジェクト構造

```
Quantitative-Text-Analysis/
├── data/
│   ├── cache/
│   └── raw/
│       ├── エクスポートされたアイテム.csv
│       └── エクスポートされたアイテム.txt
├── src/
│   └── quant_text_analysis/
│       ├── __init__.py
│       ├── __main__.py               # エントリーポイント
│       ├── cluster/
│       │   ├── __init__.py
│       │   ├── algorithms.py
│       │   └── metrics.py
│       ├── commands/
│       │   ├── __init__.py
│       │   ├── cluster_cli.py
│       │   ├── cross_table.py
│       │   ├── freq_cli.py
│       │   ├── mnlr_cli.py
│       │   └── phrases_cli.py
│       ├── config.py
│       ├── data_types.py
│       ├── features/
│       │   ├── __init__.py
│       │   ├── embeddings.py
│       │   ├── frequency.py
│       │   ├── ppmi.py
│       │   └── vocab_selection.py
│       ├── grouping.py
│       ├── io/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── writers.py
│       ├── mnlr/
│       │   ├── __init__.py
│       │   ├── coding.py
│       │   ├── model.py
│       │   ├── plotting.py
│       │   ├── statsmodels_fork.py
│       │   └── tables.py
│       ├── preprocess/
│       │   ├── __init__.py
│       │   ├── nlp_backend.py
│       │   ├── normalize.py
│       │   └── perdoc.py
│       └── settings.py
├── src/quant_text_analysis.egg-info/
├── outputs/
├── .gitignore
├── constraints.txt
├── docs.md
├── pydoc-markdown.yaml
├── pyproject.toml
├── README.md
└── requirements.txt
```

## キャッシュ機能

計算コストの高い処理結果は `data/cache/` ディレクトリにキャッシュされます：
- 文書ごとの正規化トークン列 (`per_doc_tokens_*.pkl`) ※頻度計算は軽量なため、その場で再計算します
- Truncated SVD による語埋め込み

同じデータで複数回実行する場合、キャッシュが活用され処理が高速化されます。

## 技術的特徴

- **英米表記統一**: breameライブラリによる自動的な表記統一
- **形態素解析**: spaCyによる高速で正確なトークン化
- **SVD埋め込み**: Truncated SVD による語ベクトル表現と再利用可能なキャッシュ
- **球面k-means**: L2 正規化後の語埋め込みをコサイン距離でクラスタリング
- **評価指標**: cos シルエットやクラスタ慣性を計算し、各クラスタの語ランキングを保存

## ライセンス

このプロジェクトのライセンスについては、リポジトリのライセンスファイルを参照してください。

## 貢献

バグ報告や機能提案は、GitHubのIssuesページでお願いします。

## 注意事項

- 初回実行時は、spaCyの言語モデルのダウンロードや処理に時間がかかる場合があります
- 大規模なコーパスを処理する場合は、十分なメモリが必要です
- 出力ディレクトリは実行時のタイムスタンプで自動生成されます
