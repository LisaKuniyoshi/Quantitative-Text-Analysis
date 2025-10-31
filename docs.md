# Table of Contents

* [quant\_text\_analysis.cluster.algorithms](#quant_text_analysis.cluster.algorithms)
  * [l2\_normalize\_rows](#quant_text_analysis.cluster.algorithms.l2_normalize_rows)
  * [cosine\_inertia](#quant_text_analysis.cluster.algorithms.cosine_inertia)
  * [SKMeansResult](#quant_text_analysis.cluster.algorithms.SKMeansResult)
    * [labels\_](#quant_text_analysis.cluster.algorithms.SKMeansResult.labels_)
    * [centroids\_](#quant_text_analysis.cluster.algorithms.SKMeansResult.centroids_)
    * [inertia\_](#quant_text_analysis.cluster.algorithms.SKMeansResult.inertia_)
  * [spherical\_kmeans](#quant_text_analysis.cluster.algorithms.spherical_kmeans)
* [quant\_text\_analysis.cluster.metrics](#quant_text_analysis.cluster.metrics)
  * [abstract\_cluster\_ratio](#quant_text_analysis.cluster.metrics.abstract_cluster_ratio)
* [quant\_text\_analysis.commands.cluster\_cli](#quant_text_analysis.commands.cluster_cli)
  * [main](#quant_text_analysis.commands.cluster_cli.main)
* [quant\_text\_analysis.commands.freq\_cli](#quant_text_analysis.commands.freq_cli)
  * [main](#quant_text_analysis.commands.freq_cli.main)
* [quant\_text\_analysis.commands.phrases\_cli](#quant_text_analysis.commands.phrases_cli)
  * [simple\_tokenize](#quant_text_analysis.commands.phrases_cli.simple_tokenize)
  * [build\_corpus](#quant_text_analysis.commands.phrases_cli.build_corpus)
  * [train\_phrases](#quant_text_analysis.commands.phrases_cli.train_phrases)
  * [phrase\_df\_from\_model](#quant_text_analysis.commands.phrases_cli.phrase_df_from_model)
  * [count\_phrase\_usage](#quant_text_analysis.commands.phrases_cli.count_phrase_usage)
  * [main](#quant_text_analysis.commands.phrases_cli.main)
* [quant\_text\_analysis.config](#quant_text_analysis.config)
  * [default\_columns](#quant_text_analysis.config.default_columns)
  * [default\_token\_policy](#quant_text_analysis.config.default_token_policy)
* [quant\_text\_analysis.data\_types](#quant_text_analysis.data_types)
  * [Columns](#quant_text_analysis.data_types.Columns)
  * [TokenPolicy](#quant_text_analysis.data_types.TokenPolicy)
  * [RankingParams](#quant_text_analysis.data_types.RankingParams)
  * [TokenLike](#quant_text_analysis.data_types.TokenLike)
  * [DocLike](#quant_text_analysis.data_types.DocLike)
  * [NLPBackend](#quant_text_analysis.data_types.NLPBackend)
* [quant\_text\_analysis.features.embeddings](#quant_text_analysis.features.embeddings)
  * [get\_or\_svd\_embedding](#quant_text_analysis.features.embeddings.get_or_svd_embedding)
* [quant\_text\_analysis.features.frequency](#quant_text_analysis.features.frequency)
  * [frequency\_rankings](#quant_text_analysis.features.frequency.frequency_rankings)
* [quant\_text\_analysis.features.ppmi](#quant_text_analysis.features.ppmi)
  * [EPS](#quant_text_analysis.features.ppmi.EPS)
  * [PPMIOutputs](#quant_text_analysis.features.ppmi.PPMIOutputs)
    * [vocab](#quant_text_analysis.features.ppmi.PPMIOutputs.vocab)
    * [doc\_ids](#quant_text_analysis.features.ppmi.PPMIOutputs.doc_ids)
    * [X\_tf](#quant_text_analysis.features.ppmi.PPMIOutputs.X_tf)
    * [ppmi\_word\_doc](#quant_text_analysis.features.ppmi.PPMIOutputs.ppmi_word_doc)
    * [ppmi\_word\_word](#quant_text_analysis.features.ppmi.PPMIOutputs.ppmi_word_word)
    * [cache\_key](#quant_text_analysis.features.ppmi.PPMIOutputs.cache_key)
  * [get\_or\_compute\_ppmi](#quant_text_analysis.features.ppmi.get_or_compute_ppmi)
* [quant\_text\_analysis.features.vocab\_selection](#quant_text_analysis.features.vocab_selection)
  * [build\_filtered\_tf\_matrix](#quant_text_analysis.features.vocab_selection.build_filtered_tf_matrix)
* [quant\_text\_analysis.grouping](#quant_text_analysis.grouping)
  * [period\_group\_year](#quant_text_analysis.grouping.period_group_year)
  * [method\_group](#quant_text_analysis.grouping.method_group)
* [quant\_text\_analysis.io.loader](#quant_text_analysis.io.loader)
  * [load\_df](#quant_text_analysis.io.loader.load_df)
* [quant\_text\_analysis.io.writers](#quant_text_analysis.io.writers)
  * [save\_vocab](#quant_text_analysis.io.writers.save_vocab)
  * [save\_cluster\_terms](#quant_text_analysis.io.writers.save_cluster_terms)
  * [save\_labels](#quant_text_analysis.io.writers.save_labels)
  * [save\_metrics](#quant_text_analysis.io.writers.save_metrics)
  * [save\_cluster\_ratio](#quant_text_analysis.io.writers.save_cluster_ratio)
* [quant\_text\_analysis.preprocess.nlp\_backend](#quant_text_analysis.preprocess.nlp_backend)
  * [\_SpacyTokenAdapter](#quant_text_analysis.preprocess.nlp_backend._SpacyTokenAdapter)
    * [\_\_init\_\_](#quant_text_analysis.preprocess.nlp_backend._SpacyTokenAdapter.__init__)
  * [\_SpacyDocAdapter](#quant_text_analysis.preprocess.nlp_backend._SpacyDocAdapter)
    * [\_\_init\_\_](#quant_text_analysis.preprocess.nlp_backend._SpacyDocAdapter.__init__)
    * [\_\_iter\_\_](#quant_text_analysis.preprocess.nlp_backend._SpacyDocAdapter.__iter__)
  * [SpacyBackend](#quant_text_analysis.preprocess.nlp_backend.SpacyBackend)
    * [\_\_init\_\_](#quant_text_analysis.preprocess.nlp_backend.SpacyBackend.__init__)
    * [pipe](#quant_text_analysis.preprocess.nlp_backend.SpacyBackend.pipe)
* [quant\_text\_analysis.preprocess.normalize](#quant_text_analysis.preprocess.normalize)
  * [build\_normalizer](#quant_text_analysis.preprocess.normalize.build_normalizer)
* [quant\_text\_analysis.preprocess.perdoc](#quant_text_analysis.preprocess.perdoc)
  * [analyze\_docs](#quant_text_analysis.preprocess.perdoc.analyze_docs)
  * [compute\_term\_frequencies](#quant_text_analysis.preprocess.perdoc.compute_term_frequencies)
  * [analyze\_docs\_with\_cache](#quant_text_analysis.preprocess.perdoc.analyze_docs_with_cache)
* [quant\_text\_analysis.settings](#quant_text_analysis.settings)
  * [Settings](#quant_text_analysis.settings.Settings)
    * [columns](#quant_text_analysis.settings.Settings.columns)
    * [token\_policy](#quant_text_analysis.settings.Settings.token_policy)
    * [ensure\_out\_dir](#quant_text_analysis.settings.Settings.ensure_out_dir)
* [quant\_text\_analysis.\_\_main\_\_](#quant_text_analysis.__main__)
  * [main](#quant_text_analysis.__main__.main)

<a id="quant_text_analysis.cluster.algorithms"></a>

# quant\_text\_analysis.cluster.algorithms

球面 k-means クラスタリングで利用する補助モジュール。

<a id="quant_text_analysis.cluster.algorithms.l2_normalize_rows"></a>

#### l2\_normalize\_rows

```python
def l2_normalize_rows(X: np.ndarray) -> np.ndarray
```

行方向に L2 正規化を施した行列を返す。

**Arguments**:

- `X` _numpy.ndarray_ - 正規化対象の行列。
  

**Returns**:

- `numpy.ndarray` - 各行のノルムが 1 となる行列。

<a id="quant_text_analysis.cluster.algorithms.cosine_inertia"></a>

#### cosine\_inertia

```python
def cosine_inertia(X_unit: np.ndarray, labels: np.ndarray,
                   centroids_unit: np.ndarray) -> float
```

cos 類似度に基づくクラスタリングの慣性を計算する。

**Arguments**:

- `X_unit` _numpy.ndarray_ - サンプルの単位ベクトル行列。
- `labels` _numpy.ndarray_ - 各サンプルのクラスタ割当。
- `centroids_unit` _numpy.ndarray_ - クラスタ重心の単位ベクトル行列。
  

**Returns**:

- `float` - Σ(1 - cos(x_i, μ_{label_i})) の値。

<a id="quant_text_analysis.cluster.algorithms.SKMeansResult"></a>

## SKMeansResult Objects

```python
@dataclass
class SKMeansResult()
```

spherical k-means の結果を保持するデータクラス。

**Attributes**:

- `labels_` _numpy.ndarray_ - サンプルごとのクラスタ割当。
- `centroids_` _numpy.ndarray_ - 単位ベクトルで表現されたクラスタ重心。
- `inertia_` _float_ - cos 慣性の値。

<a id="quant_text_analysis.cluster.algorithms.SKMeansResult.labels_"></a>

#### labels\_

(n_samples,)

<a id="quant_text_analysis.cluster.algorithms.SKMeansResult.centroids_"></a>

#### centroids\_

(k, d) unit

<a id="quant_text_analysis.cluster.algorithms.SKMeansResult.inertia_"></a>

#### inertia\_

Σ(1 - cos)

<a id="quant_text_analysis.cluster.algorithms.spherical_kmeans"></a>

#### spherical\_kmeans

```python
def spherical_kmeans(
        X_unit: np.ndarray,
        k: int,
        n_init: int,
        max_iter: int,
        rng: Optional[np.random.Generator] = None) -> SKMeansResult
```

cos 類似度最大化を目的とした spherical k-means を実行する。

**Arguments**:

- `X_unit` _numpy.ndarray_ - 行ごとに正規化されたサンプルベクトル。
- `k` _int_ - 生成するクラスタ数。
- `n_init` _int_ - 初期化回数。
- `max_iter` _int_ - 最大反復回数。
- `rng` _numpy.random.Generator | None_ - 乱数生成器。
  

**Returns**:

- `SKMeansResult` - 最良のクラスタリング結果。

<a id="quant_text_analysis.cluster.metrics"></a>

# quant\_text\_analysis.cluster.metrics

Cluster-level metrics computed from per-document frequency data.

<a id="quant_text_analysis.cluster.metrics.abstract_cluster_ratio"></a>

#### abstract\_cluster\_ratio

```python
def abstract_cluster_ratio(per_doc_freqs: List[PerDocFreq], vocab: List[str],
                           labels: np.ndarray) -> np.ndarray
```

文書ごとのクラスタ比率行列を算出する。

**Arguments**:

- `per_doc_freqs` _list[dict[str, float]]_ - 文書内語の確率分布。
- `vocab` _list[str]_ - 語彙リスト。
- `labels` _numpy.ndarray_ - 語彙に対応するクラスタラベル。
  

**Returns**:

- `numpy.ndarray` - 文書 × クラスタの比率行列。

<a id="quant_text_analysis.commands.cluster_cli"></a>

# quant\_text\_analysis.commands.cluster\_cli

Word clustering pipeline (PPMI → SVD → spherical k-means).

概要:
既定の設定 (`Settings`) に基づき、
要旨コーパスから語×文書 PPMI と語×語 PPMI を計算し、
SVD による語埋め込みを L2 正規化して球面 k-means でクラスタリングします。
語彙・PPMI 行列・クラスタ結果と各種メトリクスを出力します。

I/O:
読み込み:
- CSV: Settings.csv_path に指定された書誌 CSV（要旨・年・手作業タグ列）
書き込み:
- outputs/vocab.json
- outputs/PPMI_word_doc/svd_dim_{d}/top_terms_k{K}.csv
- outputs/PPMI_word_doc/svd_dim_{d}/labels_k{K}.csv
- outputs/PPMI_word_doc/svd_dim_{d}/metrics_k{K}.json
- outputs/PPMI_word_doc/svd_dim_{d}/abstract_ratio_k{K}.npy
- outputs/PPMI_word_doc/metrics.csv
- outputs/PPMI_word_word/svd_dim_{d}/top_terms_k{K}.csv
- outputs/PPMI_word_word/svd_dim_{d}/labels_k{K}.csv
- outputs/PPMI_word_word/svd_dim_{d}/metrics_k{K}.json
- outputs/PPMI_word_word/svd_dim_{d}/abstract_ratio_k{K}.npy
- outputs/PPMI_word_word/metrics.csv

設定:
すべて `Settings` で指定します（パス、`k_list`, `svd_dim`, `random_seed` など）。
形態素解析は spaCy（`spacy_model`）を使用し、文書ごとの頻度はキャッシュ可能です。

使用例:
>>> python -m quant_text_analysis.cluster.app
>>> python path/to/cluster_cli.py

<a id="quant_text_analysis.commands.cluster_cli.main"></a>

#### main

```python
def main() -> None
```

クラスタリングパイプライン全体を実行します。

注意事項:
* 乱数は `Settings.random_seed` に従います。
* キャッシュは文書頻度・PPMI・SVD の計算で利用されます。
* 語彙やクラスタリング結果は標準出力とファイルに出力されます。

<a id="quant_text_analysis.commands.freq_cli"></a>

# quant\_text\_analysis.commands.freq\_cli

Token frequency rankings for overall and predefined groups.

概要:
`Settings` に基づいて CSV を読み込み、文書内相対頻度 r(d, w) を平均化して
上位語を算出します。全体・年代・研究手法のランキングを表示し、必要に応じて
CSV に保存します。

I/O:
読み込み:
- CSV: Settings.csv_path
書き込み:
- outputs/top_words_overall.csv
- outputs/top_words_periods.csv
- outputs/top_words_methods.csv

グルーピング:
- 年代: "2014–2021" / "2022–2023" / "2024–2025"
- 手法: "qual" / "quan" / "theoretic" / "review" / "other"

使用例:
python -m quant_text_analysis.cli
python path/to/freq_cli.py

<a id="quant_text_analysis.commands.freq_cli.main"></a>

#### main

```python
def main() -> None
```

頻度ランキングを計算して表示し、必要に応じて保存します。

**Notes**:

  - `Settings.out_dir` が設定されている場合に CSV を書き出す。

<a id="quant_text_analysis.commands.phrases_cli"></a>

# quant\_text\_analysis.commands.phrases\_cli

Phrase mining with Gensim Phrases (bigrams/trigrams).

概要:
英字のみのトークン化と英米表記統一を行い、Gensim Phrases で bigram と
trigram を学習します。モデルのスコアとコーパスでの使用統計を結合し、
スコア・出現数・文書率などの指標を CSV に保存します。

I/O:
読み込み:
- CSV: Settings.csv_path（要旨列）
書き込み:
- outputs/phrases_gensim.csv

注意事項:
- Phrases 学習では `ENGLISH_CONNECTOR_WORDS` を接続語として使用します。
- 表記統一に `breame.spelling.get_american_spelling` を利用します。

使用例:
python -m quant_text_analysis.phrase_discovery
python path/to/phrases_cli.py

<a id="quant_text_analysis.commands.phrases_cli.simple_tokenize"></a>

#### simple\_tokenize

```python
def simple_tokenize(text: str) -> List[str]
```

英字のみを対象にトークン化し、表記統一を行う。

**Arguments**:

- `text` _str_ - トークン化対象の文字列。
  

**Returns**:

- `list[str]` - 小文字化・表記統一済みトークン列。

<a id="quant_text_analysis.commands.phrases_cli.build_corpus"></a>

#### build\_corpus

```python
def build_corpus(texts: Sequence[str]) -> List[List[str]]
```

文字列コーパスをトークン化済みコーポラへ変換する。

**Arguments**:

- `texts` _Sequence[str]_ - トークン化対象の文書列。
  

**Returns**:

- `list[list[str]]` - トークン化された文書ごとのトークン列。

<a id="quant_text_analysis.commands.phrases_cli.train_phrases"></a>

#### train\_phrases

```python
def train_phrases(corpus: Sequence[Sequence[str]], min_count: int,
                  threshold: float, *,
                  connector_words: Iterable[str]) -> Phrases
```

Phrases モデルを学習して返す。

**Arguments**:

- `corpus` _Sequence[Sequence[str]]_ - 学習対象のトークン化済み文書群。
- `min_count` _int_ - フレーズ抽出の最小出現回数。
- `threshold` _float_ - フレーズ候補を採用するスコア閾値。
- `connector_words` _Iterable[str]_ - 接続語として扱う語の集合。
  

**Returns**:

- `Phrases` - 学習済みの Phrases モデル。

<a id="quant_text_analysis.commands.phrases_cli.phrase_df_from_model"></a>

#### phrase\_df\_from\_model

```python
def phrase_df_from_model(model: Phrases) -> pd.DataFrame
```

Phrases モデルから候補フレーズを抽出する。

**Arguments**:

- `model` _Phrases_ - 評価対象の Phrases モデル。
  

**Returns**:

- `pandas.DataFrame` - フレーズ、語数、スコアの一覧。

<a id="quant_text_analysis.commands.phrases_cli.count_phrase_usage"></a>

#### count\_phrase\_usage

```python
def count_phrase_usage(tokenized: Sequence[Sequence[str]],
                       joiner: str = "_") -> pd.DataFrame
```

トークン列から抽出フレーズの出現統計を集計する。

**Arguments**:

- `tokenized` _Sequence[Sequence[str]]_ - トークン化済み文書群。
- `joiner` _str_ - フレーズを結合する際の区切り文字。
  

**Returns**:

- `pandas.DataFrame` - フレーズごとの出現回数・文書数・文書率。

<a id="quant_text_analysis.commands.phrases_cli.main"></a>

#### main

```python
def main() -> None
```

bigram と trigram の候補を学習し、統計を出力する。

**Notes**:

  - bi→tri の順に適用し、モデルの `export_phrases()` と実使用回数をマージする。
  - 上位候補は標準出力に表示し、全結果を CSV に保存する。

<a id="quant_text_analysis.config"></a>

# quant\_text\_analysis.config

Factory helpers for default configuration values.

<a id="quant_text_analysis.config.default_columns"></a>

#### default\_columns

```python
def default_columns() -> Columns
```

既定の列名設定を返す。

**Returns**:

- `Columns` - 既定の列名設定を表す `Columns` インスタンス。

<a id="quant_text_analysis.config.default_token_policy"></a>

#### default\_token\_policy

```python
def default_token_policy() -> TokenPolicy
```

既定のトークン正規化ポリシーを返す。

**Returns**:

- `TokenPolicy` - 既定のトークン正規化ポリシーを表す `TokenPolicy` インスタンス。

<a id="quant_text_analysis.data_types"></a>

# quant\_text\_analysis.data\_types

Shared dataclasses and protocols describing public data structures.

<a id="quant_text_analysis.data_types.Columns"></a>

## Columns Objects

```python
@dataclass(frozen=True)
class Columns()
```

分析で参照する列名をまとめた設定用データクラス。

**Attributes**:

- `abstract` _str_ - 要約テキスト列の名前。
- `year` _str_ - 発行年列の名前。
- `manual_tags` _str_ - 手動タグ列の名前。

<a id="quant_text_analysis.data_types.TokenPolicy"></a>

## TokenPolicy Objects

```python
@dataclass(frozen=True)
class TokenPolicy()
```

トークン正規化の制約を表す設定用データクラス。

**Attributes**:

- `target_pos` _frozenset[str]_ - 対象とする品詞集合。
- `exclude_ner` _frozenset[str]_ - 除外する固有表現タイプ集合。
- `exclude_propn` _bool_ - 固有名詞を除外するかどうか。
- `exclude_aux` _bool_ - 助動詞を除外するかどうか。
- `keep_surface_for` _frozenset[str]_ - 表層形を保持する品詞集合。
- `alpha_regex` _str_ - アルファベット判定に用いる正規表現。
- `forced_phrases` _tuple[tuple[str, ...], ...]_ - 強制的に多語表現とみなす語列。
- `forced_joiner` _str_ - 強制多語表現を結合する際の連結文字。
- `forced_aliases` _tuple[tuple[tuple[str, ...], str], ...]_ - 多語表現ごとの出力別名。

<a id="quant_text_analysis.data_types.RankingParams"></a>

## RankingParams Objects

```python
@dataclass(frozen=True)
class RankingParams()
```

ランキング出力時の閾値設定。

**Attributes**:

- `top_n` _int_ - 表示する上位アイテム数。
- `min_docs` _int_ - 最小出現文書数。

<a id="quant_text_analysis.data_types.TokenLike"></a>

## TokenLike Objects

```python
@runtime_checkable
class TokenLike(Protocol)
```

spaCy 互換トークンが満たすべきインターフェース。

<a id="quant_text_analysis.data_types.DocLike"></a>

## DocLike Objects

```python
@runtime_checkable
class DocLike(Protocol)
```

spaCy 互換ドキュメントが満たすべきシーケンスインターフェース。

<a id="quant_text_analysis.data_types.NLPBackend"></a>

## NLPBackend Objects

```python
@runtime_checkable
class NLPBackend(Protocol)
```

spaCy 互換 NLP モデルのインターフェース定義。

<a id="quant_text_analysis.features.embeddings"></a>

# quant\_text\_analysis.features.embeddings

<a id="quant_text_analysis.features.embeddings.get_or_svd_embedding"></a>

#### get\_or\_svd\_embedding

```python
def get_or_svd_embedding(X_wd: ArrayLike,
                         *,
                         svd_dim: int,
                         cfg: Optional[Settings] = None,
                         ppmi_cache_key: Optional[str] = None,
                         random_state: Optional[int] = None) -> np.ndarray
```

SVD 埋め込みをキャッシュから取得または計算して返す。

**Arguments**:

- `X_wd` _ArrayLike_ - 語×文書の PPMI 行列。
- `svd_dim` _int_ - 生成する埋め込み次元。
- `cfg` _Settings | None_ - 設定オブジェクト。None の場合は既定値を利用。
- `ppmi_cache_key` _str | None_ - PPMI 計算時のキャッシュキー。
- `random_state` _int | None_ - SVD の乱数シード。未指定時は設定値を使用。
  

**Returns**:

- `numpy.ndarray` - 語埋め込み行列。キャッシュ未使用時は新たに計算される。

<a id="quant_text_analysis.features.frequency"></a>

# quant\_text\_analysis.features.frequency

<a id="quant_text_analysis.features.frequency.frequency_rankings"></a>

#### frequency\_rankings

```python
def frequency_rankings(
        per_doc_freqs: List[Dict[str, float]],
        groups: Optional[List[Optional[str]]] = None
) -> Dict[str, pd.DataFrame]
```

語相対頻度をグループ別に集計しランキングを生成する。

**Arguments**:

- `per_doc_freqs` _list[dict[str, float]]_ - 文書ごとの語相対頻度分布。
- `groups` _list[str | None] | None_ - 文書が属するグループラベル。None の場合は全件を単一グループ扱い。
  

**Returns**:

  dict[str, pandas.DataFrame]: グループ ID をキーに持つランキング表。

<a id="quant_text_analysis.features.ppmi"></a>

# quant\_text\_analysis.features.ppmi

<a id="quant_text_analysis.features.ppmi.EPS"></a>

#### EPS

数値安定用（log のゼロ回避）

<a id="quant_text_analysis.features.ppmi.PPMIOutputs"></a>

## PPMIOutputs Objects

```python
@dataclass(frozen=True)
class PPMIOutputs()
```

PPMI 計算から得られる主要な成果物を保持するデータクラス。

**Attributes**:

- `vocab` _list[str]_ - 語彙リスト（行順）。
- `doc_ids` _list[int]_ - 文書 ID のリスト。
- `X_tf` _scipy.sparse.csr_matrix_ - 文書×語の正規化 TF 行列。
- `ppmi_word_doc` _scipy.sparse.csr_matrix_ - 語×文書の非対称 PPMI 行列。
- `ppmi_word_word` _scipy.sparse.csr_matrix_ - 語×語の対称 PPMI 行列。
- `cache_key` _str | None_ - キャッシュ識別子。未設定時は None。

<a id="quant_text_analysis.features.ppmi.PPMIOutputs.vocab"></a>

#### vocab

行=語の順序

<a id="quant_text_analysis.features.ppmi.PPMIOutputs.doc_ids"></a>

#### doc\_ids

0..D-1

<a id="quant_text_analysis.features.ppmi.PPMIOutputs.X_tf"></a>

#### X\_tf

D x V（正規化TF；語彙に制限後の射影行列）

<a id="quant_text_analysis.features.ppmi.PPMIOutputs.ppmi_word_doc"></a>

#### ppmi\_word\_doc

V x D（非対称PPMI；行=語）

<a id="quant_text_analysis.features.ppmi.PPMIOutputs.ppmi_word_word"></a>

#### ppmi\_word\_word

V x V（対称PPMI；行=語）

<a id="quant_text_analysis.features.ppmi.PPMIOutputs.cache_key"></a>

#### cache\_key

キャッシュ識別子（省略可）

<a id="quant_text_analysis.features.ppmi.get_or_compute_ppmi"></a>

#### get\_or\_compute\_ppmi

```python
def get_or_compute_ppmi(per_doc_freqs: List[Dict[str, float]]) -> PPMIOutputs
```

per-doc 頻度から PPMI を計算しキャッシュを活用して返す。

**Arguments**:

- `per_doc_freqs` _list[dict[str, float]]_ - 文書ごとの語相対頻度分布。
  

**Returns**:

- `PPMIOutputs` - 語彙・TF 行列・PPMI 行列のセット。キャッシュ利用時は `cache_key` を含む。

<a id="quant_text_analysis.features.vocab_selection"></a>

# quant\_text\_analysis.features.vocab\_selection

Utilities for selecting the analysis vocabulary from per-document counts.

<a id="quant_text_analysis.features.vocab_selection.build_filtered_tf_matrix"></a>

#### build\_filtered\_tf\_matrix

```python
def build_filtered_tf_matrix(per_doc_freqs: List[PerDocFreq], *, top_n: int,
                             min_docs: int) -> Tuple[sp.csr_matrix, List[str]]
```

文書ごとの語頻度辞書をベクトル化し、語彙をフィルタリングする。

**Arguments**:

- `per_doc_freqs` _list[dict[str, float]]_ - 文書ごとの語相対頻度辞書。
- `top_n` _int_ - 残す語の最大語彙数。0 以下で空語彙を返す。
- `min_docs` _int_ - 語が出現すべき最小文書数。
  

**Returns**:

  tuple[sp.csr_matrix, list[str]]: フィルタ済みの文書-語行列と語彙リスト。

<a id="quant_text_analysis.grouping"></a>

# quant\_text\_analysis.grouping

Grouping helpers used by frequency and clustering analyses.

<a id="quant_text_analysis.grouping.period_group_year"></a>

#### period\_group\_year

```python
def period_group_year(y: Optional[int]) -> Optional[str]
```

発行年から集計用の期間ラベルを生成する。

**Arguments**:

- `y` _int | None_ - 発行年。NaN もしくは None の場合は未分類とみなす。
  

**Returns**:

  str | None: 期間ラベル。該当しない場合は None。

<a id="quant_text_analysis.grouping.method_group"></a>

#### method\_group

```python
def method_group(tags: Optional[str]) -> Optional[str]
```

手動タグから研究手法カテゴリを推定する。

**Arguments**:

- `tags` _str | None_ - セミコロン区切りの手法タグ文字列。
  

**Returns**:

  str | None: 推定されたカテゴリ。該当しない場合は None。

<a id="quant_text_analysis.io.loader"></a>

# quant\_text\_analysis.io.loader

Data-loading helpers for reading prepared CSV corpora.

<a id="quant_text_analysis.io.loader.load_df"></a>

#### load\_df

```python
def load_df(csv_path: str, columns: Columns) -> pd.DataFrame
```

研究用データセットを読み込み、必要列を抽出する。

**Arguments**:

- `csv_path` _str_ - 入力 CSV ファイルのパス。
- `columns` _Columns_ - 取得対象となる列名設定。
  

**Returns**:

- `pandas.DataFrame` - 列名を標準化した DataFrame。

<a id="quant_text_analysis.io.writers"></a>

# quant\_text\_analysis.io.writers

<a id="quant_text_analysis.io.writers.save_vocab"></a>

#### save\_vocab

```python
def save_vocab(out_dir: Path, vocab: List[str]) -> None
```

語彙リストを JSON 形式で保存する。

**Arguments**:

- `out_dir` _Path_ - 出力先ディレクトリ。
- `vocab` _list[str]_ - 語彙リスト。

<a id="quant_text_analysis.io.writers.save_cluster_terms"></a>

#### save\_cluster\_terms

```python
def save_cluster_terms(out_dir: Path, k: int, vocab: List[str],
                       labels: np.ndarray,
                       silhouette_scores: np.ndarray) -> Path
```

クラスタ語リストをシルエット指標付きで CSV に保存する。

**Arguments**:

- `out_dir` _Path_ - 出力先ディレクトリ。
- `k` _int_ - クラスタ数。
- `vocab` _list[str]_ - 語彙リスト。
- `labels` _numpy.ndarray_ - 語彙に対応するクラスタラベル。
- `silhouette_scores` _numpy.ndarray_ - 各語のシルエット値。
  

**Returns**:

- `Path` - 生成された CSV ファイルのパス。

<a id="quant_text_analysis.io.writers.save_labels"></a>

#### save\_labels

```python
def save_labels(out_dir: Path, k: int, vocab: List[str],
                labels: np.ndarray) -> Path
```

語彙とクラスタ割当を CSV 形式で保存する。

**Arguments**:

- `out_dir` _Path_ - 出力先ディレクトリ。
- `k` _int_ - クラスタ数。
- `vocab` _list[str]_ - 語彙リスト。
- `labels` _numpy.ndarray_ - クラスタラベル配列。
  

**Returns**:

- `Path` - 生成された CSV ファイルのパス。

<a id="quant_text_analysis.io.writers.save_metrics"></a>

#### save\_metrics

```python
def save_metrics(out_dir: Path, k: int, *, inertia: float,
                 silhouette: Optional[float]) -> Path
```

クラスタ評価指標を JSON 形式で保存する。

**Arguments**:

- `out_dir` _Path_ - 出力先ディレクトリ。
- `k` _int_ - クラスタ数。
- `inertia` _float_ - cos 慣性。
- `silhouette` _float | None_ - cos シルエット。NaN は None に変換。
  

**Returns**:

- `Path` - 生成された JSON ファイルのパス。

<a id="quant_text_analysis.io.writers.save_cluster_ratio"></a>

#### save\_cluster\_ratio

```python
def save_cluster_ratio(out_dir: Path, k: int, M: np.ndarray) -> Path
```

文書×クラスタ比率行列を NPY 形式で保存する。

**Arguments**:

- `out_dir` _Path_ - 出力先ディレクトリ。
- `k` _int_ - クラスタ数。
- `M` _numpy.ndarray_ - 文書×クラスタ比率行列。
  

**Returns**:

- `Path` - 生成された NPY ファイルのパス。

<a id="quant_text_analysis.preprocess.nlp_backend"></a>

# quant\_text\_analysis.preprocess.nlp\_backend

<a id="quant_text_analysis.preprocess.nlp_backend._SpacyTokenAdapter"></a>

## \_SpacyTokenAdapter Objects

```python
class _SpacyTokenAdapter()
```

spaCy `Token` を薄くラップするアダプター。

<a id="quant_text_analysis.preprocess.nlp_backend._SpacyTokenAdapter.__init__"></a>

#### \_\_init\_\_

```python
def __init__(token: Token) -> None
```

アダプターを初期化する。

**Arguments**:

- `token` _Token_ - ラップ対象の spaCy トークン。

<a id="quant_text_analysis.preprocess.nlp_backend._SpacyDocAdapter"></a>

## \_SpacyDocAdapter Objects

```python
class _SpacyDocAdapter()
```

spaCy `Doc` を `TokenLike` イテレーターに変換するアダプター。

<a id="quant_text_analysis.preprocess.nlp_backend._SpacyDocAdapter.__init__"></a>

#### \_\_init\_\_

```python
def __init__(doc: Doc) -> None
```

アダプターを初期化する。

**Arguments**:

- `doc` _Doc_ - ラップ対象の spaCy ドキュメント。

<a id="quant_text_analysis.preprocess.nlp_backend._SpacyDocAdapter.__iter__"></a>

#### \_\_iter\_\_

```python
def __iter__() -> Iterator[TokenLike]
```

逐次的にトークンアダプターを生成する。

<a id="quant_text_analysis.preprocess.nlp_backend.SpacyBackend"></a>

## SpacyBackend Objects

```python
class SpacyBackend()
```

spaCy モデルを用いて文書解析を行うバックエンド。

<a id="quant_text_analysis.preprocess.nlp_backend.SpacyBackend.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model: str) -> None
```

spaCy モデルを読み込む。

**Arguments**:

- `model` _str_ - 読み込む spaCy モデル名。

<a id="quant_text_analysis.preprocess.nlp_backend.SpacyBackend.pipe"></a>

#### pipe

```python
def pipe(texts: Iterable[str]) -> Iterator[DocLike]
```

spaCy の逐次パイプラインで文書を解析する。

**Arguments**:

- `texts` _Iterable[str]_ - 解析対象のテキスト列。
  

**Returns**:

- `Iterator[DocLike]` - spaCy 互換のドキュメントイテレータ。

<a id="quant_text_analysis.preprocess.normalize"></a>

# quant\_text\_analysis.preprocess.normalize

<a id="quant_text_analysis.preprocess.normalize.build_normalizer"></a>

#### build\_normalizer

```python
def build_normalizer(policy: TokenPolicy) -> Normalizer
```

ポリシーに基づくトークン正規化関数を構築する。

**Arguments**:

- `policy` _TokenPolicy_ - 品詞・固有表現・表層保持などの条件を含む設定。
  

**Returns**:

- `Normalizer` - spaCy トークンを受け取り正規化語を返す関数。

<a id="quant_text_analysis.preprocess.perdoc"></a>

# quant\_text\_analysis.preprocess.perdoc

Per-document token analysis with caching support.

<a id="quant_text_analysis.preprocess.perdoc.analyze_docs"></a>

#### analyze\_docs

```python
def analyze_docs(backend: NLPBackend, normalizer: Normalizer,
                 texts: Sequence[str],
                 policy: TokenPolicy) -> List[PerDocTokens]
```

文書群を解析し正規化トークン列を求める。

**Arguments**:

- `backend` _NLPBackend_ - トークン化・解析を行う NLP バックエンド。
- `normalizer` _Normalizer_ - トークン正規化関数。
- `texts` _Sequence[str]_ - 解析対象の文書本文。
- `policy` _TokenPolicy_ - 強制抽出やフィルタ条件を定義するポリシー。
  

**Returns**:

- `list[list[str]]` - 文書ごとの正規化済みトークン列。

<a id="quant_text_analysis.preprocess.perdoc.compute_term_frequencies"></a>

#### compute\_term\_frequencies

```python
def compute_term_frequencies(
        per_doc_tokens: Sequence[Sequence[str]]) -> List[PerDocFreq]
```

文書ごとの正規化トークン列から相対頻度を算出する。

**Arguments**:

- `per_doc_tokens` _Sequence[Sequence[str]]_ - 文書単位の正規化トークン列。
  

**Returns**:

- `List[PerDocFreq]` - 文書内相対頻度の辞書リスト。

<a id="quant_text_analysis.preprocess.perdoc.analyze_docs_with_cache"></a>

#### analyze\_docs\_with\_cache

```python
def analyze_docs_with_cache(
        backend: NLPBackend,
        normalizer: Normalizer,
        texts: List[str],
        policy: TokenPolicy,
        *,
        cache_dir: Optional[str] = None) -> List[PerDocTokens]
```

文書解析結果（正規化トークン列）をキャッシュから取得または新規生成する。

**Arguments**:

- `backend` _NLPBackend_ - トークン化・解析を行うバックエンド。
- `normalizer` _Normalizer_ - トークン正規化に使用する関数。
- `texts` _list[str]_ - 解析対象の本文。
- `policy` _TokenPolicy_ - 強制抽出や除外条件を含むポリシー。
  

**Arguments**:

- `cache_dir` _str | None_ - トークン列を保存するキャッシュディレクトリ。
  

**Returns**:

- `List[PerDocTokens]` - 文書ごとの正規化済みトークン列。

<a id="quant_text_analysis.settings"></a>

# quant\_text\_analysis.settings

<a id="quant_text_analysis.settings.Settings"></a>

## Settings Objects

```python
@dataclass(frozen=True)
class Settings()
```

プロジェクト全体の設定値を保持する不変データクラス。

**Attributes**:

- `project_root` _Path_ - プロジェクトルートディレクトリ。
- `data_dir` _Path_ - データフォルダへのパス。
- `raw_dir` _Path_ - 生データ格納先のパス。
- `cache_dir` _Path_ - 中間生成物を保存するキャッシュディレクトリ。
- `out_dir` _Path_ - 出力ディレクトリのルート。
- `csv_path` _Path_ - 入力 CSV ファイルのパス。
- `spacy_model` _str_ - 利用する spaCy モデル名。
- `top_n` _int_ - 語彙選定時に保持する上位語数。
- `min_docs` _int_ - 語彙選定時の最小文書出現数。
- `svd_dim` _int_ - SVD における埋め込み次元。
- `k_list` _Tuple[int, ...]_ - 評価するクラスタ数の候補。
- `n_init` _int_ - k-means の初期化回数。
- `max_iter` _int_ - k-means の最大反復回数。
- `random_seed` _int_ - 乱数シード値。
- `top_words_per_cluster` _int_ - クラスタごとの上位語数。

<a id="quant_text_analysis.settings.Settings.columns"></a>

#### columns

```python
@property
def columns() -> Columns
```

使用する列名設定を返す。

**Returns**:

- `Columns` - 使用する列名設定を表す `Columns` インスタンス。

<a id="quant_text_analysis.settings.Settings.token_policy"></a>

#### token\_policy

```python
@property
def token_policy() -> TokenPolicy
```

トークン正規化の設定を返す。

**Returns**:

- `TokenPolicy` - トークン正規化に用いる `TokenPolicy` 設定。

<a id="quant_text_analysis.settings.Settings.ensure_out_dir"></a>

#### ensure\_out\_dir

```python
def ensure_out_dir() -> Path
```

出力ディレクトリを確実に作成して返す。

**Returns**:

- `Path` - 確実に存在する出力ディレクトリのパス。

<a id="quant_text_analysis.__main__"></a>

# quant\_text\_analysis.\_\_main\_\_

quant_text_analysis パッケージの CLI エントリーポイント。

<a id="quant_text_analysis.__main__.main"></a>

#### main

```python
def main(argv: List[str] | None = None) -> int
```

CLI 引数を解釈して対応するサブコマンドを実行する。

**Arguments**:

- `argv` _list[str] | None_ - サブコマンドとオプションの配列。None の場合は `sys.argv[1:]` を使用。
  

**Returns**:

- `int` - サブコマンドの終了コード。

