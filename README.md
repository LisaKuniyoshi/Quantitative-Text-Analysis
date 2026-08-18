# Quantitative Text Analysis

論文要旨を英語テキストコーパスとし、定量的なテキスト分析を行うPythonパッケージです。

## 概要

以下が主要な分析機能です：

1. **フレーズ抽出** (`phrases`) - Gensim Phrasesを用いたbigram/trigramの候補抽出
2. **頻出語分析** (`freq`) - 文書内相対頻度に基づく頻出語のランキング
3. **単語クラスタリング** (`cluster`) - 文書×語頻度を SVD で次元削減し、spherical k-means で単語をクラスタリング
4. **コード×手法クロス集計** (`cross_table`) - 文書単位でコード出現有無を集計し、研究手法カテゴリと対比
5. **ニ項ロジスティック回帰** (`mnlr`) - コード出現データをニ項ロジスティック回帰分析し、出版年および研究手法ごとの効果と手法間ペアワイズ差を評価

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
git clone https://github.com/LisaKuniyoshi/Quantitative-Text-Analysis.git
cd Quantitative-Text-Analysis
```

2. パッケージをインストールします：
```bash
pip install -e .
```

## データの準備

分析対象のデータは、以下のパスに配置してください（`settings.py`から編集可）：
- デフォルトパス: `data/raw/エクスポートされたアイテム.csv`

サンプルデータは公開していませんが、メールで連絡いただければ提供します。

CSVファイルには以下の列が必要です (Zotero のエクスポート機能を想定。出版年の区分やタグについては、`grouping.py`から編集可)：
- `abstract`: 論文の要旨（英語テキスト）
- `year`: 出版年。年代別集計では、デフォルトで 2014–2025 の範囲を 3 区分に分類する。
- `manual_tags`: 手作業でのタグ付け。セミコロン区切りで記述し、複数タグを付した場合は各カテゴリに重複集計される。デフォルトでは`qual`, `quan`, `theoretic`, `review`, `other`。

## 使用方法

### 1. フレーズ抽出 (phrases)

Gensim Phrasesを使用して、bigramとtrigramの候補を抽出します。

```bash
python -m quant_text_analysis phrases
```

**出力ファイル:**
- `outputs/{タイムスタンプ}/phrases_gensim.csv` - 抽出されたフレーズ候補とスコア

候補のうち登録するものは、`config.py`の`forced`や`forced_aliases`に追加してください。

### 2. 頻出語分析 (freq)

全体、年代別、研究手法別の頻出語ランキングを算出します。

```bash
python -m quant_text_analysis freq
```

**出力ファイル:**
- `outputs/{タイムスタンプ}/top_words_overall.csv` - 全体の頻出語
- `outputs/{タイムスタンプ}/top_words_period_{グループ名}.csv` - 年代別の頻出語
- `outputs/{タイムスタンプ}/top_words_method_{グループ名}.csv` - 研究手法別の頻出語

**年代グループ（`grouping.py`から編集可）:**
- 2014–2021
- 2022–2023
- 2024–2025

**研究手法グループ（`grouping.py`から編集可）:**
- qual (質的研究)
- quan (量的研究)
- theoretic (理論研究)
- review (レビュー研究)
- other (その他)

同一文書が複数の手法タグを持つ場合は、上記の各カテゴリに重複して集計されます。

### 3. 単語クラスタリング (cluster)

語×文書頻度行列から Truncated SVD で語埋め込みを生成し、L2 正規化後にspherical k-means でクラスタリングを実行します。

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
- クラスタ数: 23（`config.py`の`k_list` で複数指定可）
- SVD次元数: 25（`config.py`の`svd_dim_list` で複数指定可）

### 4. コード×手法クロス集計 (cross_table)

`config.CODE_MAP_CLUSTER` に定義したコードブックに基づき、各コードが各手法の論文のうち何本に出現したかのクロス集計を生成します。

```bash
python -m quant_text_analysis cross_table
```

**出力ファイル:**
- `outputs/{タイムスタンプ}/code_method_crosstab_docs.csv` - コード×研究手法（文書数）のクロス表

**特徴:**
- `config.CODE_MAP_CLUSTER` に定義したコードブックを利用（`CODE_MAP_GENDER`に編集可）
- トークン化は既存のキャッシュを再利用しつつ再計算

### 5. ニ項ロジスティック回帰と可視化 (mnlr)

文書トークンをコードに展開し、ニ項ロジスティック回帰で年・研究手法の効果を推定します。

```bash
python -m quant_text_analysis mnlr
```

**出力ファイル:**
- `CODE_MAP_CLUSTER`の分析結果
  - `outputs/{タイムスタンプ}/cluster/margeff.csv` - 各クラスタコードの回帰係数・信頼区間・Odds Ratio
  - `outputs/{タイムスタンプ}/cluster/binary_logit_summary_【コード名】.txt` - 各コードのロジット回帰要約
  - `outputs/{タイムスタンプ}/cluster/pairwise_method_tests.csv` - 手法間ペアワイズ比較の検定結果
  - `outputs/{タイムスタンプ}/cluster/odds_ratio_year.png` - 年の効果の図
  - `outputs/{タイムスタンプ}/cluster/odds_ratio_methods.png` - 手法の効果の図
  - `outputs/{タイムスタンプ}/cluster/token_counts.csv` - 各クラスタコードのトークン数サマリ
- `CODE_MAP_GENDER`の分析結果
  - `outputs/{タイムスタンプ}/gender/binlogit_summary.txt` - ジェンダーコード有無の回帰の要約
  - `outputs/{タイムスタンプ}/gender/binlogit_results.csv` - ジェンダーコード有無の回帰の推定結果
  - `outputs/{タイムスタンプ}/gender/pairwise_method_tests_binlogit.csv` - 手法間ペアワイズ比較の検定結果
  - `outputs/{タイムスタンプ}/gender/odds_ratio_methods.png` - 手法の効果の図
  - `outputs/{タイムスタンプ}/gender/year_effect_prediction.png` - 年の効果の予測プロット
  - `outputs/{タイムスタンプ}/gender/female_vs_male_summary.txt` - 女性 vs 男性比較のロジット要約
  - `outputs/{タイムスタンプ}/gender/female_vs_male_results.csv` - 女性 vs 男性比較の推定結果
  - `outputs/{タイムスタンプ}/gender/pairwise_method_tests_female_vs_male.csv` - 女性 vs 男性比較のペアワイズ検定
  - `outputs/{タイムスタンプ}/gender/female_vs_male_odds_ratio_methods.png` - 女性 vs 男性比較の手法差図
  - `outputs/{タイムスタンプ}/gender/female_vs_male_year_effect_prediction.png` - 女性 vs 男性比較の年効果予測図
  - `outputs/{タイムスタンプ}/gender/token_counts.csv` - ジェンダーコードのトークン数サマリ

**特徴:**
- 手法タグはセミコロン区切りの複数指定に対応し、`qual`/`quan`/`review`/`theoretic`
  をマルチホットなダミー変数としてモデルへ投入します。
- 文書 ID をクラスタとするロバスト共分散推定を実施
- コード定義は `config.CODE_MAP_CLUSTER` および `config.CODE_MAP_GENDER` を参照

ペアワイズ検定ユーティリティ（`quant_text_analysis.mnlr.statsmodels_fork`）は、statsmodels の公開 API のみを利用しており、平均限界効果または係数差についてロバスト共分散を尊重した検定結果を DataFrame で取得できます。

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
k_list: Tuple[int, ...] = (23,)  # クラスタ数候補
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

## 注意事項

- 初回実行時は、spaCyの言語モデルのダウンロードや処理に時間がかかる場合があります
- 大規模なコーパスを処理する場合は、十分なメモリが必要です
