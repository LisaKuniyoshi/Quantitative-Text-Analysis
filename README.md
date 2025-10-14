# Quantitative Text Analysis

論文要旨などの英語テキストコーパスに対して、定量的なテキスト分析を行うPythonパッケージです。

## 概要

本プロジェクトは、以下の3つの主要な分析機能を提供します：

1. **頻出語分析** (`freq`) - 文書内相対頻度に基づく頻出語のランキング
2. **フレーズ抽出** (`phrases`) - Gensim Phrasesを用いたbigram/trigramの候補抽出
3. **単語クラスタリング** (`cluster`) - PPMI→SVD→球面k-meansによる単語のクラスタリング

## 必要な環境

- Python 3.12.10
- 主要な依存パッケージ：
  - pandas ~=2.3
  - spacy ~=3.7
  - gensim ~=4.3
  - scikit-learn ~=1.5
  - breame ~=0.1.2

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

語×文書PPMIを計算し、SVDによる語埋め込みを経て、球面k-meansでクラスタリングを実行します。

```bash
python -m quant_text_analysis cluster
```

**出力ファイル:**
- `outputs/{タイムスタンプ}/vocab.json` - 語彙リスト
- `outputs/{タイムスタンプ}/PPMI_word_doc_VxD.npz` - 語×文書PPMI行列
- `outputs/{タイムスタンプ}/PPMI_word_word_VxV.npz` - 語×語PPMI行列
- `outputs/{タイムスタンプ}/top_terms_k{K}.csv` - 各クラスタの上位語
- `outputs/{タイムスタンプ}/labels_k{K}.csv` - 単語のクラスタラベル
- `outputs/{タイムスタンプ}/metrics_k{K}.json` - クラスタリング評価指標
- `outputs/{タイムスタンプ}/abstract_ratio_k{K}.npy` - 文書×クラスタ比率行列

**デフォルト設定:**
- クラスタ数: k=12, 16, 20
- SVD次元数: 200
- k-meansの初期化回数: 20
- 最大反復回数: 300

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
min_docs: int = 7            # 最小文書出現数

# 埋め込み
svd_dim: int = 200           # SVD次元数

# クラスタリング
k_list: Tuple[int, ...] = (12, 16, 20)  # クラスタ数候補
n_init: int = 20             # k-means初期化回数
max_iter: int = 300          # 最大反復回数
random_seed: int = 42        # 乱数シード

# 出力
top_words_per_cluster: int = 20  # クラスタごとの上位語数
```

## プロジェクト構造

```
Quantitative-Text-Analysis/
├── src/
│   └── quant_text_analysis/
│       ├── __main__.py              # エントリーポイント
│       ├── settings.py              # 設定管理
│       ├── commands/                # 各コマンドの実装
│       │   ├── freq_cli.py         # 頻出語分析
│       │   ├── phrases_cli.py      # フレーズ抽出
│       │   └── cluster_cli.py      # クラスタリング
│       ├── preprocess/              # 前処理モジュール
│       │   ├── nlp_backend.py      # spaCy処理
│       │   ├── normalize.py        # トークン正規化
│       │   └── perdoc.py           # 文書ごとの処理
│       ├── features/                # 特徴量計算
│       ├── cluster/                 # クラスタリング実装
│       ├── io/                      # 入出力処理
│       │   ├── loader.py           # データ読み込み
│       │   └── writers.py          # 結果書き出し
│       └── grouping.py              # グルーピング処理
├── data/
│   ├── raw/                         # 入力データ
│   └── cache/                       # キャッシュファイル
├── outputs/                         # 出力結果
├── pyproject.toml                   # パッケージ設定
└── requirements.txt                 # 依存パッケージ
```

## キャッシュ機能

計算コストの高い処理結果は `data/cache/` ディレクトリにキャッシュされます：
- 文書ごとの頻度情報
- PPMI行列
- SVD結果

同じデータで複数回実行する場合、キャッシュが活用され処理が高速化されます。

## 技術的特徴

- **英米表記統一**: breameライブラリによる自動的な表記統一
- **形態素解析**: spaCyによる高速で正確なトークン化
- **PPMI**: Positive Pointwise Mutual Informationによる単語共起の重み付け
- **球面k-means**: L2正規化された埋め込みベクトルに対するコサイン距離ベースのクラスタリング
- **評価指標**: シルエットスコア、Jaccard安定性など複数の指標による評価

## ライセンス

このプロジェクトのライセンスについては、リポジトリのライセンスファイルを参照してください。

## 貢献

バグ報告や機能提案は、GitHubのIssuesページでお願いします。

## 注意事項

- 初回実行時は、spaCyの言語モデルのダウンロードや処理に時間がかかる場合があります
- 大規模なコーパスを処理する場合は、十分なメモリが必要です
- 出力ディレクトリは実行時のタイムスタンプで自動生成されます
