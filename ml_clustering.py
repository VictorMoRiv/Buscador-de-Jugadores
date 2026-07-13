"""
============================================================
ML Clustering — Perfiles de juego
Liga Profesional Argentina 2025
============================================================
Modelos:
  - K-Means por posición (perfil dentro de la posición)
  - DBSCAN para detectar jugadores atípicos / outliers
  - PCA / UMAP para reducción dimensional y visualización

Output:
  data/ml/player_clusters.csv   — cluster por jugador
  data/ml/cluster_profiles.csv  — descripción de cada cluster
  data/ml/models/               — modelos serializados

Ejecutar:
    python ml_clustering.py
============================================================
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────
OUT_DIR = Path("data/ml")
MODEL_DIR = OUT_DIR / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MIN_MINUTES = 150  # filtrar jugadores con poca participación
N_COMPONENTS_PCA = 2  # dimensiones para visualización

# Features por posición
FEATURES = {
    "Portero": ["GA90", "Save%", "CS", "CS%", "SoTA", "PKsv", "minutes"],
    "Defensor": [
        "Int", "TklW", "Fls", "Crs",
        "goals", "assists", "minutes", "market_value_eur",
    ],
    "Mediocampista": [
        "goals", "assists", "TklW", "Int",
        "Fld", "Crs", "minutes", "market_value_eur",
    ],
    "Delantero": [
        "goals", "assists", "Sh", "SoT", "SoT%", "G/Sh", "G/SoT",
        "minutes", "market_value_eur",
    ],
}

# Nombres descriptivos por cluster (se calculan automáticamente pero se pueden sobreescribir)
CLUSTER_LABELS = {
    "Portero": {
        0: "Portero clasico",
        1: "Portero de salida",
        2: "Portero mixto",
    },
    "Defensor": {
        0: "Defensor defensivo puro",
        1: "Defensor mixto",
        2: "Defensor con proyeccion ofensiva",
        3: "Defensor de bajo impacto",
    },
    "Mediocampista": {
        0: "Mediocampista defensivo",
        1: "Mediocampista creador",
        2: "Mediocampista mixto",
        3: "Mediocampista ofensivo",
    },
    "Delantero": {
        0: "Delantero goleador",
        1: "Delantero asociativo",
        2: "Delantero de area",
        3: "Delantero en desarrollo",
    },
}

# Rango de clusters a evaluar por posición
K_RANGE = {
    "Portero": range(2, 4),
    "Defensor": range(3, 6),
    "Mediocampista": range(3, 6),
    "Delantero": range(2, 5),
}


def classify_pos(pos):
    p = str(pos).upper()
    # Porteros
    if "GK" in p or "POR" in p or "ARQ" in p:
        return "Portero"
    # Defensores
    if "DF" in p or "DEF" in p or "LAT" in p or "CEN" in p:
        return "Defensor"
    # Mediocampistas
    if "MF" in p or "MED" in p or "VOL" in p or "MCO" in p or "MCD" in p:
        return "Mediocampista"
    # Delanteros (Añadimos "FW", "DEL", "EXT", "DC")
    if "FW" in p or "DEL" in p or "EXT" in p or "DC" in p or "ATA" in p:
        return "Delantero"

    return "Otro"


# ══════════════════════════════════════════════
# CARGA DE DATOS
# ══════════════════════════════════════════════

def load_df() -> pd.DataFrame:
    df = pd.read_csv("master_players_final.csv")

    logger.info("🕵️‍♂️ Buscando a Milton Giménez en el CSV bruto...")
    milton_raw = df[df["player"].astype(str).str.contains("Gim", case=False, na=False)]
    if not milton_raw.empty:
        for idx, row in milton_raw.iterrows():
            logger.info(
                f"   ENCONTRADO -> Player: '{row['player']}', Team: '{row.get('team', 'N/A')}', Minutes: '{row.get('minutes', 'N/A')}', Position: '{row.get('position', 'N/A')}'"
            )
    else:
        logger.error(
            "❌ Milton Giménez NO EXISTE con ese nombre en master_players_final.csv"
        )

    # 1. Limpieza inicial de minutos
    if "minutes" in df.columns:
        df["minutes"] = df["minutes"].astype(str).str.replace(",", "").str.strip()
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)

    # 2. Limpieza de edad
    if "age" in df.columns:
        df["age_num"] = df["age"].astype(str).str.split("-").str[0]
        df["age_num"] = pd.to_numeric(df["age_num"], errors="coerce")

    # 3. PRIMERO: Convertimos absolutamente todo el bloque numérico a floats/ints limpios
    num_cols = [
        "goals", "assists", "minutes", "market_value_eur", "age_num",
        "Sh", "SoT", "SoT%", "G/Sh", "G/SoT", "Fls", "Fld", "Int",
        "TklW", "Crs", "GA", "GA90", "SoTA", "Saves", "Save%", "CS", "CS%", "PKsv"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # 4. SEGUNDO: Ahora que los datos están 100% limpios, calculamos las métricas por cada 90 minutos
    df["90s"] = df["minutes"] / 90
    s = df["90s"].replace(0, np.nan) # Evitar división por cero

    per90_cols = ["goals", "assists", "Sh", "SoT", "Fls", "Fld", "Int", "TklW", "Crs"]
    for c in per90_cols:
        if c in df.columns:
            df[c] = (df[c] / s).fillna(0).round(4) # Al final, nadie tocará estos decimales

    # 5. Clasificación y filtro final
    df["pos_group"] = df["position"].astype(str).apply(classify_pos)
    df = df[df["minutes"] >= MIN_MINUTES].reset_index(drop=True)
    logger.info(f"Jugadores con >= {MIN_MINUTES} min: {len(df)}")
    return df

# ══════════════════════════════════════════════
# SELECCIÓN ÓPTIMA DE K (silhouette)
# ══════════════════════════════════════════════


def best_k(X_scaled: np.ndarray, k_range) -> int:
    scores = {}
    for k in k_range:
        if k >= len(X_scaled):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) > 1:
            scores[k] = silhouette_score(X_scaled, labels)

    if not scores:
        return list(k_range)[0]

    best = max(scores, key=scores.get)
    logger.info(f"    Silhouette scores: {scores}")
    logger.info(f"    Mejor k={best} (score={scores[best]:.4f})")
    return best


# ══════════════════════════════════════════════
# CLUSTERING POR POSICIÓN
# ══════════════════════════════════════════════


def cluster_position(df: pd.DataFrame, pos: str) -> pd.DataFrame:
    logger.info(f"\n  Procesando: {pos}")

    subset = df[df["pos_group"] == pos].copy().reset_index(drop=True)
    feats = [f for f in FEATURES[pos] if f in subset.columns]

    if len(subset) < 5:
        logger.warning(f"  Pocos jugadores ({len(subset)}) — saltando")
        subset["cluster_kmeans"] = 0
        subset["cluster_dbscan"] = 0
        subset["cluster_label"] = "Sin cluster"
        subset["pca_x"] = 0.0
        subset["pca_y"] = 0.0
        return subset

    X = subset[feats].fillna(0).values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    if pos == "Delantero" and "goals" in feats:
            goals_idx = feats.index("goals")
            X_sc[:, goals_idx] = X_sc[:, goals_idx] * 3.0 # Le damos el triple de peso de forma limpia

    # ── K-Means ───────────────────────────────
    k = best_k(X_sc, K_RANGE[pos])
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    subset["cluster_kmeans"] = km.fit_predict(X_sc)
    logger.info(f"  K-Means k={k}: {subset['cluster_kmeans'].value_counts().to_dict()}")

    # ── DBSCAN ───────────────────────────────
    db = DBSCAN(eps=1.5, min_samples=3)
    subset["cluster_dbscan"] = db.fit_predict(X_sc)
    n_noise = (subset["cluster_dbscan"] == -1).sum()
    n_clust = subset["cluster_dbscan"].nunique() - (
        1 if -1 in subset["cluster_dbscan"].values else 0
    )
    logger.info(f"  DBSCAN: {n_clust} clusters, {n_noise} outliers")

    # ── PCA para visualización ────────────────
    pca = PCA(n_components=N_COMPONENTS_PCA, random_state=42)
    coords = pca.fit_transform(X_sc)
    subset["pca_x"] = coords[:, 0].round(4)
    subset["pca_y"] = coords[:, 1].round(4)
    var_explained = pca.explained_variance_ratio_.sum() * 100
    logger.info(f"  PCA varianza explicada: {var_explained:.1f}%")

    # ── Labels descriptivos ───────────────────
    label_map = CLUSTER_LABELS.get(pos, {})

    # Auto-generar labels basados en stats dominantes del centroide
    for c in subset["cluster_kmeans"].unique():
        if c not in label_map:
            members = subset[subset["cluster_kmeans"] == c][feats].mean()
            top_feat = members.idxmax()
            label_map[c] = f"{pos} — alto {top_feat}"

    subset["cluster_label"] = (
        subset["cluster_kmeans"].map(label_map).fillna(f"{pos} otro")
    )

    # ── Guardar scaler y modelo ───────────────
    joblib.dump(scaler, MODEL_DIR / f"scaler_{pos}.pkl")
    joblib.dump(km, MODEL_DIR / f"kmeans_{pos}.pkl")
    joblib.dump(pca, MODEL_DIR / f"pca_{pos}.pkl")
    joblib.dump(feats, MODEL_DIR / f"feats_{pos}.pkl")

    return subset


# ══════════════════════════════════════════════
# PERFILES DE CLUSTER
# ══════════════════════════════════════════════


def build_cluster_profiles(df_all: pd.DataFrame) -> pd.DataFrame:
    """Calcula stats promedio por cluster para describir cada perfil."""
    rows = []
    stat_cols = [
        "goals",
        "assists",
        "minutes",
        "market_value_eur",
        "Sh",
        "SoT",
        "Int",
        "TklW",
        "Save%",
        "GA90",
        "CS",
    ]
    stat_cols = [c for c in stat_cols if c in df_all.columns]

    for (pos, cluster), grp in df_all.groupby(["pos_group", "cluster_kmeans"]):
        label = grp["cluster_label"].iloc[0]
        row = {
            "pos_group": pos,
            "cluster_id": int(cluster),
            "cluster_label": label,
            "n_players": len(grp),
            "players_sample": ", ".join(grp.nlargest(3, "minutes")["player"].tolist()),
        }
        for c in stat_cols:
            if c in grp.columns:
                row[f"avg_{c}"] = round(grp[c].mean(), 2)
        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 55)
    logger.info("  ML Clustering — Liga Profesional Argentina 2025")
    logger.info("=" * 55)

    df = load_df()

    all_results = []
    for pos in ["Portero", "Defensor", "Mediocampista", "Delantero"]:
        result = cluster_position(df, pos)
        all_results.append(result)

    df_all = pd.concat(all_results, ignore_index=True)

    # Guardar dataset con clusters
    out_players = OUT_DIR / "player_clusters.csv"
    df_all.to_csv(out_players, index=False, encoding="utf-8-sig")
    logger.success(f"✓ player_clusters.csv guardado ({len(df_all)} jugadores)")

    # Guardar perfiles
    profiles = build_cluster_profiles(df_all)
    out_prof = OUT_DIR / "cluster_profiles.csv"
    profiles.to_csv(out_prof, index=False, encoding="utf-8-sig")
    logger.success(f"✓ cluster_profiles.csv guardado ({len(profiles)} clusters)")

    # Resumen
    print("\n" + "=" * 55)
    print("  RESUMEN DE CLUSTERS")
    print("=" * 55)
    for pos in ["Portero", "Defensor", "Mediocampista", "Delantero"]:
        pos_df = df_all[df_all["pos_group"] == pos]
        print(f"\n{pos} ({len(pos_df)} jugadores):")
        for label, grp in pos_df.groupby("cluster_label"):
            sample = ", ".join(grp.nlargest(2, "minutes")["player"].tolist())
            print(f"  [{grp['cluster_kmeans'].iloc[0]}] {label} — {len(grp)} jugadores")
            print(f"      Ejemplo: {sample}")

    print(f"\n✅ Modelos guardados en: {MODEL_DIR}/")
    print("   Siguiente paso: streamlit run app.py")
