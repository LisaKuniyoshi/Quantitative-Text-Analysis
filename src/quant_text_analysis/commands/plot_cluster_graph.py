import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D


# ==== 設定 =========================================================
CSV_PATH = r"C:\Users\Lisa\Documents\大学\卒論\Quantitative-Text-Analysis\outputs\save\20251116_191033_cluster\svd_dim_25\centroid_cosine_distances_k23.csv"
SIMILARITY_THRESHOLD = 0.5
OUTPUT_PATH = r"C:\Users\Lisa\Documents\大学\卒論\Quantitative-Text-Analysis\outputs\save\20251116_191033_cluster\svd_dim_25\cluster_network_selected_bw.png"
FIGSIZE = (5, 5)

# フォント設定（環境に合わせて "Yu Mincho" / "游明朝" / "MS Mincho" などに変更）
FONT_FAMILY = "Yu Mincho"   # or "游明朝"
BASE_FONTSIZE = 10.5

plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["font.size"] = BASE_FONTSIZE

SELECTED_CLUSTERS = [
    "cluster_1", "cluster_2", "cluster_4", "cluster_5",
    "cluster_8", "cluster_10", "cluster_11", "cluster_18",
    "cluster_19", "cluster_21", "cluster_22",
]

nickname_map = {
    "cluster_1": "抑圧",
    "cluster_2": "戦略",
    "cluster_4": "手段",
    "cluster_5": "女性の見逃し",
    "cluster_8": "自殺リスク",
    "cluster_10": "疲弊",
    "cluster_11": "社会的欲求",
    "cluster_18": "認知-行動ギャップ",
    "cluster_19": "子供の性差",
    "cluster_21": "診断時期",
    "cluster_22": "正式診断",
}
# ==================================================================


def build_graph_from_distance_matrix(df_dist, threshold=0.3, selected=None):
    if selected is not None:
        df_dist = df_dist.loc[selected, selected]

    df_sim = 1.0 - df_dist
    G = nx.Graph()

    for node in df_sim.index:
        G.add_node(node)

    labels = list(df_sim.index)
    n = len(labels)

    for i in range(n):
        for j in range(i + 1, n):
            src = labels[i]
            dst = labels[j]
            sim = df_sim.iloc[i, j]

            if pd.isna(sim):
                continue
            if sim >= threshold:
                G.add_edge(src, dst, weight=sim)

    return G


def draw_graph(G, output_path, title=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D  # 凡例用

    plt.figure(figsize=FIGSIZE)

    # spring_layoutで少しゆとりを持たせる（k は広がり具合）
    pos = nx.spring_layout(G, weight="weight", k=5.0, iterations=300, seed=6)

    # ノード：ごく薄い灰色に黒枠
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=900,
        node_color="#f5f5f5",  # 薄いグレー
        linewidths=0.8,
        edgecolors="black",
    )

    legend_handles = None  # 後で凡例を作るためのハンドル

    # エッジ：min–max正規化で太さに強弱（白黒なので色は黒のみ）
    if len(G.edges()) > 0:
        raw_weights = [G[u][v]["weight"] for u, v in G.edges()]
        w_min = min(raw_weights)
        w_max = max(raw_weights)

        width_min = 1.0
        width_max = 5.0

        if w_max > w_min:
            widths = [
                width_min + (w - w_min) / (w_max - w_min) * (width_max - width_min)
                for w in raw_weights
            ]
        else:
            widths = [(width_min + width_max) / 2.0 for _ in raw_weights]

        nx.draw_networkx_edges(
            G,
            pos,
            width=widths,
            alpha=0.9,
            edge_color="black",
        )

        # --- エッジ太さの凡例を作成 ------------------------------------
        # 代表値として「最小・中央値・最大」の3本を示す
        w_mid = 0.5 * (w_min + w_max)
        rep_weights = [0.5, 0.6, 0.7]
        rep_labels = ["0.50", "0.60", "0.70"]

        rep_widths = []
        if w_max > w_min:
            for w in rep_weights:
                rep_widths.append(
                    width_min + (w - w_min) / (w_max - w_min) * (width_max - width_min)
                )
        else:
            rep_widths = [(width_min + width_max) / 2.0] * len(rep_weights)

        legend_handles = [
            Line2D(
                [], [],
                color="black",
                linewidth=lw,
                label=lab,
            )
            for lw, lab in zip(rep_widths, rep_labels)
        ]
        # -------------------------------------------------------------

    # ラベル：日本語ニックネーム・游明朝・太字（白背景付き）
    labels = {n: nickname_map.get(n, n) for n in G.nodes()}
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=BASE_FONTSIZE,
        font_family=FONT_FAMILY,
        font_weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.2",
            fc="white",   # 背景色
            ec="none",
            alpha=1.0,
        ),
    )

    if title:
        plt.title(title, fontsize=BASE_FONTSIZE)

    # 凡例を配置（位置は必要に応じて変更: upper right / lower left など）
    if legend_handles is not None:
        plt.legend(
            handles=legend_handles,
            title="重心間のコサイン類似度",
            loc="lower right",
            frameon=True,
            fontsize=BASE_FONTSIZE,
            title_fontsize=BASE_FONTSIZE,
        )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    # plt.show()

def main():
    df_dist = pd.read_csv(CSV_PATH, index_col=0)

    G = build_graph_from_distance_matrix(
        df_dist,
        threshold=SIMILARITY_THRESHOLD,
        selected=SELECTED_CLUSTERS,
    )

    draw_graph(G, OUTPUT_PATH)


if __name__ == "__main__":
    main()
