"""
============================================================
ML Recommender — Recomendador basado en clustering
============================================================
Dado un jugador de referencia, encuentra los candidatos
más similares dentro del mismo cluster K-Means.

Ranking dentro del cluster:
  1. Mismo cluster K-Means (mismo perfil de juego)
  2. Score de similitud dentro del cluster (distancia al centroide)
  3. Filtros opcionales: edad, valor, minutos
============================================================
"""

import unicodedata
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import euclidean_distances

warnings.filterwarnings("ignore")

MODEL_DIR = Path("data/ml/models")
CLUSTER_CSV = Path("data/ml/player_clusters.csv")


def _normalize_text(text: str) -> str:
    """Remueve acentos, espacios extra y pasa a minúsculas."""
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    return text


class MLRecommender:
    """Recomendador de jugadores basado en clustering K-Means."""

    def __init__(self):
        self.df_clusters = None
        self.models = {}  # {pos: {scaler, kmeans, pca, feats}}
        self._load()

    def _load(self):
        if not CLUSTER_CSV.exists():
            logger.error(
                f"No se encontró {CLUSTER_CSV} — ejecuta ml_clustering.py primero"
            )
            return

        self.df_clusters = pd.read_csv(CLUSTER_CSV)

        # OPTIMIZACIÓN: Crear columna normalizada una sola vez al cargar los datos
        self.df_clusters["player_norm"] = self.df_clusters["player"].apply(
            _normalize_text
        )

        logger.info(f"Clusters cargados: {len(self.df_clusters)} jugadores")

        for pos in ["Portero", "Defensor", "Mediocampista", "Delantero"]:
            try:
                self.models[pos] = {
                    "scaler": joblib.load(MODEL_DIR / f"scaler_{pos}.pkl"),
                    "kmeans": joblib.load(MODEL_DIR / f"kmeans_{pos}.pkl"),
                    "pca": joblib.load(MODEL_DIR / f"pca_{pos}.pkl"),
                    "feats": joblib.load(MODEL_DIR / f"feats_{pos}.pkl"),
                }
                logger.info(f"  Modelo {pos} cargado")
            except Exception as e:
                logger.warning(f"  No se pudo cargar modelo {pos}: {e}")

    @property
    def available(self) -> bool:
        return self.df_clusters is not None and len(self.models) > 0

    def get_player_cluster(self, player: str) -> dict | None:
        """Devuelve info del cluster de un jugador usando el nombre normalizado."""
        if self.df_clusters is None:
            return None

        search_name = _normalize_text(player)
        row = self.df_clusters[self.df_clusters["player_norm"] == search_name]

        if row.empty:
            return None

        r = row.iloc[0]
        return {
            "player": r["player"],
            "team": r["team"],
            "pos_group": r["pos_group"],
            "cluster_kmeans": int(r["cluster_kmeans"]),
            "cluster_label": r["cluster_label"],
            "cluster_dbscan": int(r["cluster_dbscan"]),
            "pca_x": float(r["pca_x"]),
            "pca_y": float(r["pca_y"]),
        }

    def recommend(
            self,
            ref_player: str,
            exclude_team: str,
            n: int = 6,
            max_age: float = 35,
            min_minutes: float = 200,
            max_value_eur: float = None,
        ) -> pd.DataFrame:
            """Recomienda jugadores similares al jugador de referencia."""
            if not self.available:
                return pd.DataFrame()

            # Info del jugador de referencia (ya usa búsqueda normalizada)
            ref_info = self.get_player_cluster(ref_player)
            if ref_info is None:
                logger.warning(f"Jugador no encontrado en clusters: {ref_player}")
                return pd.DataFrame()

            pos = ref_info["pos_group"]
            cluster = ref_info["cluster_kmeans"]

            # Usamos el nombre real exacto que encontramos en la base de datos para excluirlo
            actual_ref_name = ref_info["player"]

            # Pool: mismo cluster, diferente equipo
            pool = self.df_clusters[
                (self.df_clusters["pos_group"] == pos)
                & (self.df_clusters["cluster_kmeans"] == cluster)
                & (self.df_clusters["team"] != exclude_team)
                & (self.df_clusters["player"] != actual_ref_name)
            ].copy()

            # Filtros opcionales
            if "age_num" in pool.columns:
                pool = pool[pool["age_num"] <= max_age]
            if "minutes" in pool.columns:
                pool = pool[pool["minutes"] >= min_minutes]
            if max_value_eur and "market_value_eur" in pool.columns:
                pool = pool[
                    (pool["market_value_eur"] <= max_value_eur)
                    | (pool["market_value_eur"] == 0)
                ]

            if pool.empty:
                # Fallback: mismo pos, buscar en otros clusters
                logger.info(f"  Sin candidatos en cluster {cluster} — expandiendo búsqueda")
                pool = self.df_clusters[
                    (self.df_clusters["pos_group"] == pos)
                    & (self.df_clusters["team"] != exclude_team)
                    & (self.df_clusters["player"] != actual_ref_name)
                ].copy()

            if pool.empty:
                return pd.DataFrame()

            # 1. Cargar el transformador y las columnas asignadas a la posición
            scaler = self.models[pos]["scaler"]
            feats = self.models[pos]["feats"]

            # 2. Extraer las estadísticas de la base de datos para el jugador y el pool
            ref_row = self.df_clusters[self.df_clusters["player"] == actual_ref_name].iloc[0]
            ref_stats = np.array([ref_row[feats].fillna(0).values])
            pool_stats = pool[feats].fillna(0).values

            # 3. Estandarizar usando el StandardScaler original
            ref_scaled = scaler.transform(ref_stats)
            pool_scaled = scaler.transform(pool_stats)

            # ✨ 4. Replicar la ponderación de goles para delanteros (Sincronía con el clustering)
            if pos == "Delantero" and "goals" in feats:
                goals_idx = feats.index("goals")
                ref_scaled[:, goals_idx] = ref_scaled[:, goals_idx] * 3.0
                pool_scaled[:, goals_idx] = pool_scaled[:, goals_idx] * 3.0

            # 5. Medir las distancias euclidianas en el espacio real de alta dimensión
            distances = euclidean_distances(ref_scaled, pool_scaled)[0]

            pool["pca_distance"] = distances # Mantenemos el nombre original para compatibilidad con JS/Streamlit
            gamma = 1.0
            pool["similitud_ml"]= (np.exp(-gamma * distances)*100).round(1)

            result = pool.nsmallest(n, "pca_distance").copy()
            result["mismo_cluster"] = result["cluster_kmeans"] == cluster

            # Eliminar columna temporal antes de retornar para no ensuciar el output
            if "player_norm" in result.columns:
                result = result.drop(columns=["player_norm"])

            return result


    def cluster_scatter_data(self, pos: str) -> pd.DataFrame:
        """Devuelve datos PCA de todos los jugadores de una posición para scatter."""
        if self.df_clusters is None:
            return pd.DataFrame()
        data = self.df_clusters[self.df_clusters["pos_group"] == pos].copy()
        if "player_norm" in data.columns:
            data = data.drop(columns=["player_norm"])
        return data

    def cluster_profile(self, pos: str, cluster_id: int) -> dict:
        """Devuelve stats promedio del cluster."""
        if self.df_clusters is None:
            return {}
        grp = self.df_clusters[
            (self.df_clusters["pos_group"] == pos) &
            (self.df_clusters["cluster_kmeans"] == cluster_id)
        ]
        if grp.empty:
            return {}
        stat_cols = ["goals","assists","minutes","market_value_eur",
                     "Sh","SoT","Int","TklW","Save%","GA90","CS"]
        stat_cols = [c for c in stat_cols if c in grp.columns]
        profile = {
            "label":   grp["cluster_label"].iloc[0],
            "n":       len(grp),
            "players": grp.nlargest(5,"minutes")["player"].tolist(),
        }
        for c in stat_cols:
            profile[f"avg_{c}"] = round(grp[c].mean(), 2)
        return profile


# Test rápido
if __name__ == "__main__":
    rec = MLRecommender()
    if rec.available:
        test_player = "Milton Giménez"
        test_team = "Boca Juniors"

        print(f"\nRecomendaciones para reemplazar a {test_player} ({test_team}):")
        recs = rec.recommend(test_player, exclude_team=test_team, n=6)

        if not recs.empty:
            print(
                recs[
                    [
                        "player",
                        "team",
                        "cluster_label",
                        "similitud_ml",
                        "goals",
                        "assists",
                    ]
                ].to_string(index=False)
            )
        else:
            print("Sin recomendaciones")

        info = rec.get_player_cluster(test_player)
        print(f"\nCluster de {test_player}: {info}")
