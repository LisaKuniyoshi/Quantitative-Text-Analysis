import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# ------------------------------------------------------------
# ユーザーが指定するパラメータ
# ------------------------------------------------------------
DATA_DIR = "C:\\Users\\Lisa\\Documents\\大学\\卒論\\Quantitative-Text-Analysis\\outputs\\save\\20251020_000722_cluster\\svd_dim_25"     # cluster_terms_k{k}.csv が置かれているディレクトリ
METRICS_CSV = "C:\\Users\\Lisa\\Documents\\大学\\卒論\\Quantitative-Text-Analysis\\outputs\\save\\20251020_000722_cluster\\metrics.csv"  # metrics.csv のパス
OUTPUT_DIR = "C:\\Users\\Lisa\\Documents\\大学\\卒論\\Quantitative-Text-Analysis\\outputs\\save\\20251020_000722_cluster\\silhouette_plots"  # 画像を保存したいフォルダ
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 可視化したい k のリスト
range_k = list(range(2, 40))   # または例: [3, 10, 15, 20]


# ------------------------------------------------------------
# メトリクス（平均シルエット係数）を読み込む
# ------------------------------------------------------------
df_metrics = pd.read_csv(METRICS_CSV)
# metrics.csv は B列が k, C列が silhouette_avg という仕様
df_metrics.columns = ["dim", "k", "silhouette_avg"]
metrics_dict = df_metrics.set_index("k")["silhouette_avg"].to_dict()


# ------------------------------------------------------------
# 各 k について silhouette plot を描画
# ------------------------------------------------------------
for k in range_k:

    csv_path = os.path.join(DATA_DIR, f"cluster_terms_k{k}.csv")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path} (skip)")
        continue

    df = pd.read_csv(csv_path)
    # A列: cluster, B列: term, C列: silhouette
    df.columns = ["cluster", "term", "silhouette"]

    # プロット準備
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(10, 7)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(df) + (k + 1) * 10])

    # 平均シルエット係数
    silhouette_avg = metrics_dict.get(k, None)

    y_lower = 10
    for i in range(k):
        # クラスター i の silhouette 値を抽出
        ith = df[df["cluster"] == i]["silhouette"].values
        ith.sort()

        size_i = len(ith)
        y_upper = y_lower + size_i

        color = cm.nipy_spectral(float(i) / k)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax1.text(-0.05, y_lower + 0.5 * size_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title(f"Silhouette plot for k = {k}")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    if silhouette_avg is not None:
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"silhouette_k{k}.png")
    fig.savefig(out_path, dpi=300)

    plt.close(fig)