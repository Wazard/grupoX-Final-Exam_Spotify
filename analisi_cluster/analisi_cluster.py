# ============================================================
# KMeans clustering con PCA prima del clustering (PCA solo per KMeans)
# + scelta k + valutazione + profilo + descrizione (a quantili) + grafici (PCA / t-SNE / UMAP)
# ============================================================

import os
import warnings
from dataclasses import dataclass
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



warnings.filterwarnings("ignore")


# -------------------------
# CONFIG
# -------------------------
@dataclass
class Settings:
    data_path: str = r"c:\Users\Gianmario\Desktop\Python\Esercitazioni\Corso_GiGroup\grupoX-Final-Exam_Spotify\data/processed/tracks_processed_normalized.csv"
    clustered_path: str = "tracks_with_clusters.csv"

    # feature numeriche per clustering musicale
    features: tuple = (
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "loudness",
        "danceability"
    )

    random_state: int = 42
    n_init: int = 10

    # range k per la selezione
    k_min: int = 2
    k_max: int = 30

    # campionamento per t-SNE/UMAP
    max_points_viz: int = 5000

    # PCA: usata SOLO per creare lo "spazio" su cui KMeans clusterizza
    use_pca_for_clustering: bool = True
    # tieni il 90% della varianza (puoi mettere anche un intero tipo 5)
    pca_n_components: Union[float, int] = 0.90


CFG = Settings()


# -------------------------
# UTILS
# -------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File non trovato: {path}")
    return pd.read_csv(path)


def get_X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Mancano queste feature nel dataset: {missing}")

    X = df[features].copy()

    if X.isna().any().any():
        before = len(X)
        X = X.dropna()
        after = len(X)
        print(f"⚠️ Trovati NaN: droppate {before - after} righe per clustering/metriche.")

    return X


def maybe_sample(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=random_state)


def build_cluster_space(X: pd.DataFrame, cfg: Settings) -> Tuple[np.ndarray, Optional[PCA]]:
    """
    Crea lo spazio su cui KMeans lavora.
    - Se cfg.use_pca_for_clustering=True: PCA(X) -> Xc
    - Altrimenti: X.values -> Xc
    NOTA: Questo NON modifica il dataset salvato: PCA serve solo a KMeans.
    """
    if not cfg.use_pca_for_clustering:
        return X.values, None

    pca = PCA(n_components=cfg.pca_n_components, random_state=cfg.random_state)
    Xc = pca.fit_transform(X.values)

    tot = float(np.sum(pca.explained_variance_ratio_))
    ncomp = int(getattr(pca, "n_components_", Xc.shape[1]))
    print(f"🧩 PCA per clustering attiva | componenti={ncomp} | varianza spiegata={tot:.3f}")

    return Xc, pca


# -------------------------
# 1) SCELTA DI K (metriche calcolate nello stesso spazio usato da KMeans)
# -------------------------
def evaluate_k_range(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    X = get_X(df, list(cfg.features))
    Xc, _ = build_cluster_space(X, cfg)

    rows = []
    for k in range(cfg.k_min, cfg.k_max + 1):
        kmeans = MiniBatchKMeans(
                                            n_clusters=k,
                                            random_state=cfg.random_state,
                                            batch_size=2048,
                                            n_init=5,          # per k_select non serve alto
                                        )

        labels = kmeans.fit_predict(Xc)

        inertia = float(kmeans.inertia_)
        sil = float(silhouette_score(Xc, labels))
        ch = float(calinski_harabasz_score(Xc, labels))
        db = float(davies_bouldin_score(Xc, labels))

        rows.append(
            {
                "k": k,
                "inertia": inertia,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
            }
        )

        print(f"k={k:2d} | inertia={inertia:,.2f} | sil={sil:.3f} | CH={ch:,.1f} | DB={db:.3f}")

    results = pd.DataFrame(rows)

    best_k_sil = int(results.loc[results["silhouette"].idxmax(), "k"])
    print("\n✅ Suggerimento (massimo silhouette): k =", best_k_sil)

    r = results.copy()
    r["sil_norm"] = (r["silhouette"] - r["silhouette"].min()) / (r["silhouette"].max() - r["silhouette"].min() + 1e-9)
    r["db_norm"] = (r["davies_bouldin"] - r["davies_bouldin"].min()) / (r["davies_bouldin"].max() - r["davies_bouldin"].min() + 1e-9)
    r["score_tradeoff"] = r["sil_norm"] - r["db_norm"]  # alto è meglio
    best_k_trade = int(r.loc[r["score_tradeoff"].idxmax(), "k"])
    print("✅ Suggerimento (trade-off silhouette alto / DB basso): k =", best_k_trade)

    return results


def plot_k_selection(results: pd.DataFrame) -> None:
    plt.figure()
    plt.plot(results["k"], results["inertia"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia (Elbow)")
    plt.title("Elbow Method")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(results["k"], results["silhouette"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.title("Silhouette vs k")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(results["k"], results["davies_bouldin"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Davies-Bouldin (lower is better)")
    plt.title("Davies-Bouldin vs k")
    plt.tight_layout()
    plt.show()


# -------------------------
# 2) TRAIN + SALVA (salva dataset originale + colonna cluster, PCA NON viene salvata)
# -------------------------
def train_kmeans_and_save(df: pd.DataFrame, cfg: Settings, k: int) -> pd.DataFrame:
    X = get_X(df, list(cfg.features))
    Xc, _ = build_cluster_space(X, cfg)

    kmeans = MiniBatchKMeans(
                                    n_clusters=k,
                                    random_state=cfg.random_state,
                                    batch_size=2048,
                                    n_init=10,         # qui puoi tenere più stabilità
                                )

    labels = kmeans.fit_predict(Xc)

    df_out = df.loc[X.index].copy()
    df_out["cluster"] = labels
    df_out.to_csv(cfg.clustered_path, index=False)

    print(f"\n✔ Salvato: {cfg.clustered_path}")
    return df_out


# -------------------------
# 3) VALUTAZIONE QUALITÀ (metriche nello spazio PCA usato per KMeans)
# -------------------------
def evaluate_clustering_quality(df_clustered: pd.DataFrame, cfg: Settings) -> dict:
    if "cluster" not in df_clustered.columns:
        raise ValueError("Manca la colonna 'cluster'.")

    X = get_X(df_clustered, list(cfg.features))
    Xc, _ = build_cluster_space(X, cfg)
    labels = df_clustered.loc[X.index, "cluster"].values

    sil = float(silhouette_score(Xc, labels))
    ch = float(calinski_harabasz_score(Xc, labels))
    db = float(davies_bouldin_score(Xc, labels))

    out = {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}

    print("\n📈 QUALITÀ CLUSTER (calcolata nello stesso spazio usato per clustering)")
    print(f"Silhouette        : {sil:.3f} (più alto è meglio)")
    print(f"Calinski-Harabasz : {ch:,.1f} (più alto è meglio)")
    print(f"Davies-Bouldin    : {db:.3f} (più basso è meglio)")

    return out


# -------------------------
# 4) PROFILO + DESCRIZIONE (robusta su dataset standardizzato: usa quantili, non threshold fissi)
# -------------------------
def cluster_profile(df_clustered: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    return df_clustered.groupby("cluster")[list(cfg.features)].mean()


def describe_clusters_from_profile(profile: pd.DataFrame, df_reference: pd.DataFrame, cfg: Settings) -> None:
    """
    Descrizione cluster robusta per dati normalizzati (z-score o simile).
    Invece di soglie fisse (0.7/0.3), usa quantili globali del dataset.
    """
    Xref = get_X(df_reference, list(cfg.features))

    q25 = Xref.quantile(0.25)
    q75 = Xref.quantile(0.75)

    def tag_by_quantiles(feat: str, value: float) -> Optional[str]:
        if feat not in q25.index:
            return None
        if value >= q75[feat]:
            return f"{feat} alto"
        if value <= q25[feat]:
            return f"{feat} basso"
        return None

    print("\n🧠 DESCRIZIONE AUTOMATICA DEI CLUSTER (basata su quantili globali)")
    for cid, row in profile.iterrows():
        tags = []

        # Tag “semantici” (mappati a feature)
        mapping = {
            "energy": ("alta energia", "bassa energia"),
            "acousticness": ("più acustico", "meno acustico"),
            "instrumentalness": ("più strumentale", "meno strumentale"),
            "speechiness": ("più parlato/rap", "meno parlato"),
            "valence": ("mood più positivo", "mood più cupo"),
            "tempo": ("più veloce", "più lento"),
            "loudness": ("più loud", "più soft"),
        }

        for feat, (hi_tag, lo_tag) in mapping.items():
            if feat not in row.index:
                continue
            if row[feat] >= q75[feat]:
                tags.append(hi_tag)
            elif row[feat] <= q25[feat]:
                tags.append(lo_tag)

        if not tags:
            tags = ["profilo intermedio"]

        print(f"Cluster {cid}: " + ", ".join(tags))


# -------------------------
# 5) VISUALIZZAZIONI (PCA 2D / t-SNE / UMAP) sulle feature originali
# -------------------------
def plot_pca_2d(df_clustered: pd.DataFrame, cfg: Settings) -> None:
    X = get_X(df_clustered, list(cfg.features))
    labels = df_clustered.loc[X.index, "cluster"].values

    pca = PCA(n_components=2, random_state=cfg.random_state)
    X2 = pca.fit_transform(X.values)

    var = pca.explained_variance_ratio_
    print(f"\nPCA 2D var explained: PC1={var[0]:.3f}, PC2={var[1]:.3f} (tot={var.sum():.3f})")

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=10)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clusters in PCA 2D (visual)")
    plt.tight_layout()
    plt.show()


def plot_tsne(df_clustered: pd.DataFrame, cfg: Settings, perplexity: float = 30.0) -> None:
    df_s = maybe_sample(df_clustered, cfg.max_points_viz, cfg.random_state)
    X = get_X(df_s, list(cfg.features))
    labels = df_s.loc[X.index, "cluster"].values

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=cfg.random_state,
        init="pca",
        learning_rate="auto",
    )
    X2 = tsne.fit_transform(X.values)

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=10)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(f"Clusters in t-SNE 2D (sample={len(X)})")
    plt.tight_layout()
    plt.show()


def plot_umap(df_clustered: pd.DataFrame, cfg: Settings, n_neighbors: int = 15, min_dist: float = 0.1) -> None:
    try:
        import umap  # type: ignore
    except Exception:
        print("\n❌ UMAP non disponibile. Installa con: pip install umap-learn")
        return

    df_s = maybe_sample(df_clustered, cfg.max_points_viz, cfg.random_state)
    X = get_X(df_s, list(cfg.features))
    labels = df_s.loc[X.index, "cluster"].values

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=cfg.random_state,
    )
    X2 = reducer.fit_transform(X.values)

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=10)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(f"Clusters in UMAP 2D (sample={len(X)})")
    plt.tight_layout()
    plt.show()

# -------------------------
# 6) CONFRONTO CLUSTER vs GENERE
# -------------------------
def cluster_genre_table(df_clustered: pd.DataFrame, genre_col: str = "track_genre") -> pd.DataFrame:
    if "cluster" not in df_clustered.columns:
        raise ValueError("Manca la colonna 'cluster'.")
    if genre_col not in df_clustered.columns:
        raise ValueError(f"Manca la colonna genere '{genre_col}' nel dataset.")
    return pd.crosstab(df_clustered["cluster"], df_clustered[genre_col])


def top_genres_per_cluster(df_clustered: pd.DataFrame, genre_col: str = "track_genre", top_n: int = 5) -> None:
    if "cluster" not in df_clustered.columns:
        raise ValueError("Manca la colonna 'cluster'.")
    if genre_col not in df_clustered.columns:
        raise ValueError(f"Manca la colonna genere '{genre_col}' nel dataset.")

    print("\n🎵 TOP GENERI PER CLUSTER")
    for cid, g in df_clustered.groupby("cluster"):
        top = g[genre_col].value_counts(normalize=True).head(top_n)
        print(f"\nCluster {cid} (n={len(g)})")
        for genre, share in top.items():
            print(f"  {str(genre):<25s} {share:.1%}")


def cluster_purity_entropy(df_clustered: pd.DataFrame, genre_col: str = "track_genre") -> pd.DataFrame:
    if "cluster" not in df_clustered.columns:
        raise ValueError("Manca la colonna 'cluster'.")
    if genre_col not in df_clustered.columns:
        raise ValueError(f"Manca la colonna genere '{genre_col}' nel dataset.")

    rows = []
    for cid, g in df_clustered.groupby("cluster"):
        counts = g[genre_col].value_counts()
        probs = counts / counts.sum()

        purity = float(probs.max())  # quota del genere dominante
        ent = float(entropy(probs))  # Shannon entropy (log naturale)

        rows.append({
            "cluster": cid,
            "size": len(g),
            "purity": purity,
            "entropy": ent,
            "n_genres_in_cluster": int((counts > 0).sum())
        })

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    # MODES:
    #   "k_select" -> metriche per diversi k + grafici
    #   "train"    -> allena KMeans (su PCA-space) e salva tracks_with_clusters.csv (con feature originali + cluster)
    #   "analyze"  -> profilo + descrizione (quantili) + metriche qualità (richiede file cluster salvato)
    #   "viz"      -> grafici PCA + t-SNE + (opzionale) UMAP (richiede file cluster salvato)
    #   "cluster_vs_genre" -> 
    MODE ="cluster_vs_genre"

    # per MODE="train": k scelto
    K_CHOSEN = 10

    # per MODE="viz"
    DO_PCA = True
    DO_TSNE = True
    DO_UMAP = False

    if MODE == "k_select":
        df = load_data(CFG.data_path)
        results = evaluate_k_range(df, CFG)
        plot_k_selection(results)

    elif MODE == "train":
        df = load_data(CFG.data_path)
        df_clustered = train_kmeans_and_save(df, CFG, k=K_CHOSEN)

        prof = cluster_profile(df_clustered, CFG)
        print("\n📊 PROFILO (mean) DEI CLUSTER")
        print(prof)

    elif MODE == "analyze":
        df_original = load_data(CFG.data_path)
        df_clustered = load_data(CFG.clustered_path)

        _ = evaluate_clustering_quality(df_clustered, CFG)

        prof = cluster_profile(df_clustered, CFG)
        print("\n📊 PROFILO (mean) DEI CLUSTER")
        print(prof)

        describe_clusters_from_profile(prof, df_reference=df_original, cfg=CFG)

    elif MODE == "viz":
        df_clustered = load_data(CFG.clustered_path)

        if DO_PCA:
            plot_pca_2d(df_clustered, CFG)
        if DO_TSNE:
            plot_tsne(df_clustered, CFG, perplexity=30.0)
        if DO_UMAP:
            plot_umap(df_clustered, CFG, n_neighbors=15, min_dist=0.1)

    elif MODE == "cluster_vs_genre":
        from sklearn.metrics import normalized_mutual_info_score
        from scipy.stats import entropy
        df_clustered = load_data(CFG.clustered_path)

        table = cluster_genre_table(df_clustered)
        print(table)

        top_genres_per_cluster(df_clustered, top_n=5)

        pe = cluster_purity_entropy(df_clustered)
        print(pe)

        nmi = normalized_mutual_info_score(
            df_clustered["track_genre"],
            df_clustered["cluster"]
        )
        print(f"NMI (cluster vs genre): {nmi:.3f}")
    
    

    else:
        raise ValueError(f"MODE non valido: {MODE}")

