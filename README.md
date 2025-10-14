# Quantitative Text Analysis

テキストデータに対する定量的分析を行うPythonパッケージです。学術論文の要旨などを対象に、頻出語抽出・フレーズマイニング・クラスタリングを実行できます。

## 概要

本パッケージは、大量のテキストデータから有意義な情報を抽出するための3つの主要な分析機能を提供します：

1. **頻出語分析 (freq)**: 文書内相対頻度を平均化し、全体・年代別・研究手法別の上位語を算出
2. **フレーズマイニング (phrases)**: Gensim Phrasesを使用したbigram/trigramの自動抽出
3. **語クラスタリング (cluster)**: PPMI（正の点ごと相互情報量）→ SVD → 球面k-meansによる語の意味的グループ化

## 主な特徴

- **spaCyベースの形態素解析**: 英語テキストの高精度なトークン化
- **キャッシュ機構**: 計算コストの高い前処理結果を再利用
- **柔軟な設定**: `Settings`クラスで全てのパラメータを一元管理
- **豊富な出力**: CSV・JSON・NumPy配列による結果保存
- **グループ別分析**: 年代や研究手法などの属性でサブグループ分析が可能

## 必要要件

- Python 3.12.10
- 主要な依存パッケージ:
  - pandas ~= 2.3
  - spacy ~= 3.7 (en_core_web_sm モデル)
  - gensim ~= 4.3
  - scikit-learn ~= 1.5
  - breame ~= 0.1.2

## インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/Slaine00/Quantitative-Text-Analysis.git
cd Quantitative-Text-Analysis
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

または、開発モードでインストール：

```bash
pip install -e .
```

## プロジェクト構成

```
Quantitative-Text-Analysis/
├── src/
│   └── quant_text_analysis/
│       ├── __main__.py          # CLIエントリーポイント
│       ├── settings.py          # 設定管理
│       ├── commands/            # サブコマンド実装
│       │   ├── freq_cli.py      # 頻出語分析
│       │   ├── phrases_cli.py   # フレーズマイニング
│       │   └── cluster_cli.py   # クラスタリング
│       ├── preprocess/          # テキスト前処理
│       ├── features/            # 特徴量抽出
│       ├── cluster/             # クラスタリングロジック
│       └── io/                  # データ入出力
├── data/
│   ├── raw/                     # 入力CSVファイル
│   └── cache/                   # 中間生成物キャッシュ
├── outputs/                     # 分析結果出力先
├── pyproject.toml
└── requirements.txt
```

## 使い方

### データの準備

入力CSVファイルを `data/raw/` ディレクトリに配置してください。デフォルトでは `エクスポートされたアイテム.csv` という名前のファイルが期待されます。

CSVファイルには以下の列が必要です：
- `abstract`: 論文要旨などのテキストデータ
- `year`: 年代グループ化用の年情報
- `manual_tags`: 研究手法などのタグ情報

### 1. 頻出語分析

文書内相対頻度を平均化し、上位語をランキング表示します。

```bash
python -m quant_text_analysis freq
```

**出力ファイル:**
- `outputs/top_words_overall.csv`: 全体の上位語
- `outputs/top_words_period_{グループ名}.csv`: 年代別上位語
- `outputs/top_words_method_{グループ名}.csv`: 研究手法別上位語

**グループ分類:**
- 年代: "2014–2021" / "2022–2023" / "2024–2025"
- 手法: "qual" / "quan" / "theoretic" / "review" / "other"

### 2. フレーズマイニング

Gensim Phrasesでbigram/trigramを学習し、候補フレーズを抽出します。

```bash
python -m quant_text_analysis phrases
```

**出力ファイル:**
- `outputs/phrases_gensim.csv`: フレーズ候補とスコア

**特徴:**
- 英字トークン化と英米表記統一（breameライブラリ使用）
- 接続語（"of", "and" など）を考慮したフレーズ学習
- モデル由来スコアと実コーパスでの使用統計を結合

### 3. 語クラスタリング

PPMI行列をSVDで次元削減し、球面k-meansでクラスタリングします。

```bash
python -m quant_text_analysis cluster
```

**出力ファイル:**
- `outputs/vocab.json`: 語彙リスト
- `outputs/PPMI_word_doc_VxD.npz`: 語×文書PPMI行列
- `outputs/PPMI_word_word_VxV.npz`: 語×語PPMI行列
- `outputs/top_terms_k{K}.csv`: クラスタごとの上位語
- `outputs/labels_k{K}.csv`: 各語のクラスタラベル
- `outputs/metrics_k{K}.json`: シルエット係数などの評価指標
- `outputs/abstract_ratio_k{K}.npy`: 文書×クラスタ比率

**パラメータ（Settings内で設定）:**
- `k_list`: クラスタ数の候補（デフォルト: 12, 16, 20）
- `svd_dim`: SVD次元数（デフォルト: 200）
- `n_init`: k-means初期化回数（デフォルト: 20）
- `random_seed`: 乱数シード（デフォルト: 42）

## 設定のカスタマイズ

`src/quant_text_analysis/settings.py` で各種パラメータを変更できます：

```python
@dataclass(frozen=True)
class Settings:
    # パス設定
    csv_path: Path = RAW_DIR / "エクスポートされたアイテム.csv"
    
    # テキスト前処理
    spacy_model: str = "en_core_web_sm"
    
    # 語彙選定
    top_n: int = 10_000        # 保持する上位語数
    min_docs: int = 7          # 最小文書出現数
    
    # 埋め込み次元
    svd_dim: int = 200
    
    # クラスタリング
    k_list: Tuple[int, ...] = (12, 16, 20)
    n_init: int = 20
    max_iter: int = 300
    random_seed: int = 42
    
    # 出力
    top_words_per_cluster: int = 20
```

## 技術詳細

### テキスト前処理パイプライン

1. **トークン化**: spaCyによる形態素解析
2. **正規化**: 小文字化、ストップワード除去、品詞フィルタリング
3. **表記統一**: breameによる英米表記統一
4. **キャッシュ**: per-doc頻度を再利用

### PPMI行列の計算

- 語×文書の共起行列から非対称PPMIを計算
- 対称的な語×語PPMI行列も同時に生成
- スパース行列形式で効率的に保存

### 球面k-means

1. SVDで語埋め込みを獲得
2. L2正規化で単位球面上に射影
3. コサイン類似度ベースのk-meansでクラスタリング
4. シルエット係数とJaccard安定性で評価

## 注意事項

- 初回実行時はspaCyモデルのダウンロードと前処理に時間がかかります
- キャッシュファイルは `data/cache/` に保存され、再実行時に再利用されます
- 出力ディレクトリは実行時刻ごとに `outputs/YYYYMMDD_HHMMSS/` 形式で作成されます

## トラブルシューティング

### spaCyモデルが見つからない場合

```bash
python -m spacy download en_core_web_sm
```

### メモリ不足エラーが発生する場合

`Settings`で`top_n`や`svd_dim`を小さくしてください。

### CSVファイルの文字コードエラー

入力CSVファイルがUTF-8でエンコードされていることを確認してください。

## ライセンス

このプロジェクトのライセンスについては、リポジトリのLICENSEファイルを参照してください。

## 貢献

バグ報告や機能追加の提案は、GitHubのIssuesでお願いします。

## 開発者向け情報

### ドキュメント生成

```bash
pdoc3 --html --output-dir docs src/quant_text_analysis
```

### テスト実行

（テストフレームワークが追加された場合）

```bash
pytest tests/
```

## 参考文献

- [Gensim Phrases](https://radimrehurek.com/gensim/models/phrases.html)
- [spaCy](https://spacy.io/)
- [PPMI (Positive Pointwise Mutual Information)](https://en.wikipedia.org/wiki/Pointwise_mutual_information)
- [Spherical k-means](https://en.wikipedia.org/wiki/K-means_clustering#Spherical_k-means_clustering)
