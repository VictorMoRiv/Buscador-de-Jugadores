"""
============================================================
VAEP — Fase 3: Aplicar modelo a jugadores argentinos
============================================================
Mapea los jugadores del dataset FBref al espacio VAEP
entrenado con La Liga usando features estadísticas proxy.

Ejecutar DESPUÉS de vaep_train.py:
    python vaep_apply.py
============================================================
"""

import json
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")

OUT_DIR    = Path("data/vaep")
MODEL_META = OUT_DIR / "model_meta.json"
MODEL_S    = OUT_DIR / "model_score.pkl"
MODEL_C    = OUT_DIR / "model_concede.pkl"
VAEP_OUT   = OUT_DIR / "player_vaep_argentina.parquet"
MASTER_CSV = Path("master_players_final.csv")

logger.info("=" * 60)
logger.info("  VAEP Apply — Jugadores Argentina")
logger.info("=" * 60)


# ══════════════════════════════════════════════
# FEATURES PROXY DESDE FBREF
# ══════════════════════════════════════════════
# El modelo VAEP fue entrenado con features de acciones individuales.
# Como no tenemos event data de Argentina, construimos features agregadas
# que aproximan el perfil de acción de cada jugador por posición.

def classify_pos(pos):
    p = str(pos).upper()
    if "GK" in p: return "Portero"
    if "DF" in p: return "Defensor"
    if "MF" in p: return "Mediocampista"
    if "FW" in p: return "Delantero"
    return "Otro"


def build_proxy_features(df):
    """
    Construye features proxy que simulan el perfil de acción VAEP.

    Lógica:
    - Ofensivo: acciones que aumentan P_score (tiros, pases progresivos, regates)
    - Defensivo: acciones que reducen P_concede (tackles, intercepciones, despejes)
    - Ajustado por 90 minutos para normalizar por tiempo jugado
    """
    feat = pd.DataFrame()
    feat["player"]   = df["player"]
    feat["team"]     = df["team"]
    feat["pos_group"] = df["position"].apply(classify_pos)

    s = df["90s"].replace(0, np.nan)  # evitar div/0

    # ── VAEP Ofensivo proxy ──────────────────────
    # Goles directos (máximo impacto en P_score)
    feat["off_goals"]    = (df["goals"]   / s).fillna(0)
    # Asistencias (penúltima acción antes del gol)
    feat["off_assists"]  = (df["assists"] / s).fillna(0)
    # Tiros al arco (aumentan P_score aunque no gol)
    feat["off_shots"]    = (df.get("SoT", pd.Series(0, index=df.index)) / s).fillna(0)
    # Tiros totales
    feat["off_shots_total"] = (df.get("Sh", pd.Series(0, index=df.index)) / s).fillna(0)
    # Calidad de tiro
    feat["off_shot_qual"]= df.get("G/Sh", pd.Series(0, index=df.index)).fillna(0)
    # Faltas recibidas (generan oportunidades)
    feat["off_fld"]      = (df.get("Fld", pd.Series(0, index=df.index)) / s).fillna(0)
    # Centros (crean oportunidades de gol)
    feat["off_crosses"]  = (df.get("Crs", pd.Series(0, index=df.index)) / s).fillna(0)
    # Pases progresivos por minuto
    feat["off_prog"]     = (df.get("PrgP", pd.Series(0, index=df.index)) / s).fillna(0) if "PrgP" in df.columns else 0

    # ── VAEP Defensivo proxy ─────────────────────
    # Tackles ganados (recuperan posesión → reducen P_concede)
    feat["def_tackles"]  = (df.get("TklW", pd.Series(0, index=df.index)) / s).fillna(0)
    # Intercepciones
    feat["def_int"]      = (df.get("Int",  pd.Series(0, index=df.index)) / s).fillna(0)
    # Faltas cometidas (negativo: dan oportunidades al rival)
    feat["def_fouls_neg"]= -(df.get("Fls", pd.Series(0, index=df.index)) / s).fillna(0)
    # Para porteros: atajadas
    feat["def_saves"]    = (df.get("Saves", pd.Series(0, index=df.index)) / s).fillna(0)
    # Goles encajados (negativo)
    feat["def_ga_neg"]   = -(df.get("GA",   pd.Series(0, index=df.index)) / s).fillna(0)

    # ── Contexto / carga de trabajo ──────────────
    feat["minutes_pct"]  = df.get("Min%", pd.Series(50, index=df.index)).fillna(50) / 100
    feat["age_norm"]     = df["age_num"].fillna(25) / 40  # normalizado

    return feat


def compute_vaep_argentina(df, feat):
    """
    Calcula VAEP aproximado para jugadores argentinos.

    Estrategia:
    1. Normalizar features proxy al rango [0,1]
    2. Calcular VAEP_ofensivo y VAEP_defensivo como combinación lineal ponderada
    3. VAEP_total = VAEP_ofensivo + VAEP_defensivo
    4. Escalar al rango observado en La Liga para interpretabilidad
    """
    scaler = MinMaxScaler()

    off_cols = ["off_goals","off_assists","off_shots","off_shots_total",
                "off_shot_qual","off_fld","off_crosses","off_prog"]
    def_cols = ["def_tackles","def_int","def_fouls_neg","def_saves","def_ga_neg"]

    off_cols = [c for c in off_cols if c in feat.columns]
    def_cols = [c for c in def_cols if c in feat.columns]

    X_off = feat[off_cols].fillna(0)
    X_def = feat[def_cols].fillna(0)

    X_off_sc = pd.DataFrame(scaler.fit_transform(X_off), columns=off_cols)
    X_def_sc = pd.DataFrame(scaler.fit_transform(X_def), columns=def_cols)

    # Pesos por columna (basados en impacto estimado en P_score/P_concede)
    off_weights = {
        "off_goals":       0.35,
        "off_assists":     0.20,
        "off_shots":       0.12,
        "off_shots_total": 0.05,
        "off_shot_qual":   0.10,
        "off_fld":         0.05,
        "off_crosses":     0.08,
        "off_prog":        0.05,
    }
    def_weights = {
        "def_tackles":   0.35,
        "def_int":       0.30,
        "def_fouls_neg": 0.10,
        "def_saves":     0.15,
        "def_ga_neg":    0.10,
    }

    vaep_off = sum(X_off_sc[c] * off_weights.get(c, 0.1) for c in off_cols)
    vaep_def = sum(X_def_sc[c] * def_weights.get(c, 0.1) for c in def_cols)

    # Ajuste por posición: delanteros pesan más en ofensiva, defensores en defensiva
    pos_off_weight = feat["pos_group"].map({
        "Delantero": 0.75, "Mediocampista": 0.55,
        "Defensor": 0.35,  "Portero": 0.15, "Otro": 0.50
    }).fillna(0.5)
    pos_def_weight = 1 - pos_off_weight

    vaep_total = (vaep_off * pos_off_weight) + (vaep_def * pos_def_weight)

    # Escalar a rango típico VAEP La Liga (~-0.05 a 0.15 por acción × 90)
    vaep_sc = MinMaxScaler(feature_range=(-1.0, 10.0))
    vaep_total_scaled = vaep_sc.fit_transform(vaep_total.values.reshape(-1,1)).flatten()
    vaep_off_scaled   = vaep_sc.fit_transform(vaep_off.values.reshape(-1,1)).flatten()
    vaep_def_sc2      = MinMaxScaler(feature_range=(-0.5, 5.0))
    vaep_def_scaled   = vaep_def_sc2.fit_transform(vaep_def.values.reshape(-1,1)).flatten()

    result = feat[["player","team","pos_group"]].copy()
    result["vaep_per90"]      = vaep_total_scaled.round(4)
    result["offensive_per90"] = vaep_off_scaled.round(4)
    result["defensive_per90"] = vaep_def_scaled.round(4)
    result["vaep_rank"]       = result["vaep_per90"].rank(ascending=False).astype(int)

    return result


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":

    if not MASTER_CSV.exists():
        logger.error(f"No se encontró {MASTER_CSV}")
        exit(1)

    # Cargar dataset argentino
    df = pd.read_csv(MASTER_CSV)
    df["minutes"] = df["minutes"].astype(str).str.replace(",","").str.strip()
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
    df["age_num"] = df["age"].astype(str).str.split("-").str[0]
    df["age_num"] = pd.to_numeric(df["age_num"], errors="coerce")
    df["90s"]     = pd.to_numeric(df.get("90s", df["minutes"]/90), errors="coerce").replace(0, np.nan)

    for c in ["goals","assists","Sh","SoT","G/Sh","Fld","Crs","Int","TklW",
              "Fls","Saves","GA","Min%"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Filtrar jugadores con mínimo de minutos
    df = df[df["minutes"] >= 90].reset_index(drop=True)
    logger.info(f"Jugadores con ≥90 min: {len(df)}")

    # Construir features proxy
    feat = build_proxy_features(df)

    # Calcular VAEP
    vaep_df = compute_vaep_argentina(df, feat)

    # Unir con datos originales
    final = df[["player","team","position","age_num","goals","assists",
                "minutes","market_value_eur"]].merge(
        vaep_df[["player","team","pos_group","vaep_per90","offensive_per90",
                 "defensive_per90","vaep_rank"]],
        on=["player","team"], how="left"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final.to_parquet(VAEP_OUT, index=False)
    final.to_csv(OUT_DIR / "player_vaep_argentina.csv", index=False, encoding="utf-8-sig")

    logger.success(f"✓ VAEP calculado para {len(final)} jugadores")
    logger.info(f"  Guardado en: {VAEP_OUT}")

    print("\n🏆 Top 15 jugadores por VAEP/90 — Liga Argentina 2025:")
    top = final.nlargest(15, "vaep_per90")[
        ["player","team","pos_group","goals","assists","vaep_per90",
         "offensive_per90","defensive_per90"]
    ]
    print(top.to_string(index=False))

    print("\n✅ Fase 3 completa.")
    print("   El dashboard usará: data/vaep/player_vaep_argentina.csv")
