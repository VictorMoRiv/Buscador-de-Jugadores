"""
VAEP Training — La Liga 2020/21 (StatsBomb)
socceraction 1.5.3 — labels construidos manualmente
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
warnings.filterwarnings("ignore")

try:
    from socceraction.data.statsbomb import StatsBombLoader
    import socceraction.spadl as spadl
    import socceraction.spadl.statsbomb as spadlsb
    import socceraction.vaep.formula as vaepformula
    import socceraction.vaep.features as vaepfeatures
    import joblib
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)

# ── Config ────────────────────────────────────
COMPETITION_ID = 11
SEASON_ID      = 90
MAX_MATCHES    = None
NB_PREV        = 3   # ventana de acciones previas
NB_NEXT        = 10  # ventana para calcular labels

OUT_DIR = Path("data/vaep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SCORE   = OUT_DIR / "model_score.pkl"
MODEL_CONCEDE = OUT_DIR / "model_concede.pkl"
SPADL_PATH    = OUT_DIR / "spadl_actions.parquet"
FEATURES_PATH = OUT_DIR / "vaep_features.parquet"
LABELS_PATH   = OUT_DIR / "vaep_labels.parquet"

FEATURE_FUNCS = [
    vaepfeatures.actiontype_onehot,
    vaepfeatures.result_onehot,
    vaepfeatures.actiontype_result_onehot,
    vaepfeatures.goalscore,
    vaepfeatures.startlocation,
    vaepfeatures.endlocation,
    vaepfeatures.movement,
    vaepfeatures.space_delta,
    vaepfeatures.startpolar,
    vaepfeatures.endpolar,
    vaepfeatures.team,
    vaepfeatures.time_delta,
]

logger.info("=" * 55)
logger.info("  VAEP Training — La Liga 2020/21")
logger.info("=" * 55)


def apply_features(states: list) -> pd.DataFrame:
    parts = []
    for func in FEATURE_FUNCS:
        try:
            parts.append(func(states))
        except Exception as e:
            logger.debug(f"  {func.__name__} falló: {e}")
    return pd.concat(parts, axis=1) if parts else pd.DataFrame()


def compute_labels(actions: pd.DataFrame, nb_next: int = NB_NEXT) -> pd.DataFrame:
    """
    Construye labels manualmente:
    - scores:   ¿el equipo actual marca gol en las próximas nb_next acciones?
    - concedes: ¿el equipo actual recibe gol en las próximas nb_next acciones?
    """
    # Acción de gol: type_id == 11 (shot) con result_id == 1 (success)
    # En SPADL: type_name='shot' o 'shot_penalty' o 'shot_freekick' + result_name='success'
    cfg = spadl.config

    # IDs de acciones de gol (tiro exitoso)
    goal_type_ids = [
        cfg.actiontypes.index(a)
        for a in ["shot", "shot_penalty", "shot_freekick"]
        if a in cfg.actiontypes
    ]
    success_id = cfg.results.index("success") if "success" in cfg.results else 1

    n = len(actions)
    scores   = np.zeros(n, dtype=bool)
    concedes = np.zeros(n, dtype=bool)

    team_col = "team_id" if "team_id" in actions.columns else "team_name"

    for i in range(n):
        current_team = actions[team_col].iloc[i]
        window = actions.iloc[i+1 : i+1+nb_next]
        if window.empty:
            continue

        is_goal = (
            window["type_id"].isin(goal_type_ids) &
            (window["result_id"] == success_id)
        ) if "type_id" in window.columns else pd.Series(False, index=window.index)

        same_team  = window[team_col] == current_team
        other_team = window[team_col] != current_team

        scores[i]   = (is_goal & same_team).any()
        concedes[i] = (is_goal & other_team).any()

    return pd.DataFrame({"scores": scores, "concedes": concedes})


# ══════════════════════════════════════════════
# FASE 1
# ══════════════════════════════════════════════

def download_and_convert() -> pd.DataFrame:
    if SPADL_PATH.exists():
        logger.info("SPADL ya existe — cargando")
        return pd.read_parquet(SPADL_PATH)

    loader = StatsBombLoader(getter="remote", creds={"user": None, "passwd": None})
    games  = loader.games(COMPETITION_ID, SEASON_ID)
    logger.info(f"Partidos: {len(games)}")
    if MAX_MATCHES:
        games = games.head(MAX_MATCHES)

    all_actions, errors = [], 0
    for i, (_, game) in enumerate(games.iterrows(), 1):
        gid = game["game_id"]
        try:
            events  = loader.events(gid)
            actions = spadlsb.convert_to_actions(events, game["home_team_id"])
            actions["game_id"] = gid
            all_actions.append(actions)
            if i % 10 == 0 or i == 1:
                logger.info(f"  [{i}/{len(games)}] game {gid} — {len(actions)} acc.")
        except Exception as e:
            errors += 1
            logger.warning(f"  Error {gid}: {e}")

    df = pd.concat(all_actions, ignore_index=True)
    df.to_parquet(SPADL_PATH, index=False)
    logger.success(f"✓ {len(df):,} acciones ({errors} errores)")
    return df


# ══════════════════════════════════════════════
# FASE 2A
# ══════════════════════════════════════════════

def compute_features_labels(df_actions: pd.DataFrame):
    if FEATURES_PATH.exists() and LABELS_PATH.exists():
        logger.info("Features/labels ya existen — cargando")
        return pd.read_parquet(FEATURES_PATH), pd.read_parquet(LABELS_PATH)

    logger.info("Calculando features y labels...")
    all_X, all_y = [], []
    games = df_actions["game_id"].unique()

    for i, gid in enumerate(games, 1):
        group = df_actions[df_actions["game_id"] == gid].reset_index(drop=True)
        try:
            named  = spadl.add_names(group)
            states = vaepfeatures.gamestates(named, nb_prev_actions=NB_PREV)
            X_game = apply_features(states)
            y_game = compute_labels(group)

            n = min(len(X_game), len(y_game))
            all_X.append(X_game.iloc[:n])
            all_y.append(y_game.iloc[:n])

            if i % 10 == 0 or i == 1:
                logger.info(f"  [{i}/{len(games)}] game {gid} — {n} samples")

        except Exception as e:
            logger.debug(f"  Error game {gid}: {e}")

    if not all_X:
        logger.error("No se pudieron calcular features.")
        return None, None

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)
    X.to_parquet(FEATURES_PATH, index=False)
    y.to_parquet(LABELS_PATH,   index=False)
    logger.success(f"✓ Features: {X.shape} | Labels: {y.shape}")
    logger.info(f"  % goles (scores):   {y['scores'].mean()*100:.2f}%")
    logger.info(f"  % goles (concedes): {y['concedes'].mean()*100:.2f}%")
    return X, y


# ══════════════════════════════════════════════
# FASE 2B
# ══════════════════════════════════════════════

def train_models(X: pd.DataFrame, y: pd.DataFrame):
    if MODEL_SCORE.exists() and MODEL_CONCEDE.exists():
        logger.info("Modelos ya existen — cargando")
        return joblib.load(MODEL_SCORE), joblib.load(MODEL_CONCEDE)

    logger.info("Entrenando XGBoost...")
    Xc = X.select_dtypes(include=[np.number]).fillna(0)
    n  = int(len(Xc) * 0.8)
    Xtr, Xte = Xc.iloc[:n], Xc.iloc[n:]
    ytr, yte = y.iloc[:n],   y.iloc[n:]

    p = dict(n_estimators=100, max_depth=5, learning_rate=0.1,
             subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)

    logger.info("  P_score...")
    ms = xgb.XGBClassifier(**p)
    ms.fit(Xtr, ytr["scores"].astype(int),
           eval_set=[(Xte, yte["scores"].astype(int))], verbose=False)
    auc_s = roc_auc_score(yte["scores"], ms.predict_proba(Xte)[:,1])
    logger.success(f"  ✓ AUC P_score: {auc_s:.4f}")

    logger.info("  P_concede...")
    mc = xgb.XGBClassifier(**p)
    mc.fit(Xtr, ytr["concedes"].astype(int),
           eval_set=[(Xte, yte["concedes"].astype(int))], verbose=False)
    auc_c = roc_auc_score(yte["concedes"], mc.predict_proba(Xte)[:,1])
    logger.success(f"  ✓ AUC P_concede: {auc_c:.4f}")

    joblib.dump(ms, MODEL_SCORE)
    joblib.dump(mc, MODEL_CONCEDE)
    with open(OUT_DIR / "model_meta.json", "w") as f:
        json.dump({"auc_score": auc_s, "auc_concede": auc_c,
                   "feature_cols": Xc.columns.tolist()}, f, indent=2)
    logger.success(f"✓ Modelos en {OUT_DIR}/")
    return ms, mc


# ══════════════════════════════════════════════
# FASE 2C
# ══════════════════════════════════════════════

def compute_player_vaep(df_actions: pd.DataFrame, ms, mc) -> pd.DataFrame:
    out = OUT_DIR / "player_vaep_laliga.parquet"
    if out.exists():
        logger.info("VAEP La Liga ya calculado")
        return pd.read_parquet(out)

    logger.info("Calculando VAEP por jugador...")
    rows  = []
    games = df_actions["game_id"].unique()

    for gid in games:
        group = df_actions[df_actions["game_id"] == gid].reset_index(drop=True)
        try:
            named  = spadl.add_names(group)
            states = vaepfeatures.gamestates(named, nb_prev_actions=NB_PREV)
            Xm     = apply_features(states).select_dtypes(include=[np.number]).fillna(0)

            n         = min(len(Xm), len(named))
            Pscores   = pd.Series(ms.predict_proba(Xm)[:n, 1])
            Pconcedes = pd.Series(mc.predict_proba(Xm)[:n, 1])

            # vaepformula.value requiere add_names (type_name, result_name, etc.)
            vals = vaepformula.value(named.iloc[:n], Pscores, Pconcedes)

            grp = named.iloc[:n].copy()
            grp["vaep_value"]      = vals["vaep_value"].values
            grp["offensive_value"] = vals["offensive_value"].values
            grp["defensive_value"] = vals["defensive_value"].values

            cols = [c for c in ["player_id","player_name","team_id",
                                 "vaep_value","offensive_value","defensive_value"]
                    if c in grp.columns]
            rows.append(grp[cols])
        except Exception as e:
            logger.debug(f"  Error {gid}: {e}")

    if not rows:
        return pd.DataFrame()

    dv = pd.concat(rows, ignore_index=True)
    gc = [c for c in ["player_id","player_name","team_id"] if c in dv.columns]
    pv = dv.groupby(gc).agg(
        vaep_total      =("vaep_value","sum"),
        offensive_total =("offensive_value","sum"),
        defensive_total =("defensive_value","sum"),
        n_actions       =("vaep_value","count"),
    ).reset_index()

    pv["vaep_per90"]      = (pv["vaep_total"]      / pv["n_actions"] * 90).round(4)
    pv["offensive_per90"] = (pv["offensive_total"]  / pv["n_actions"] * 90).round(4)
    pv["defensive_per90"] = (pv["defensive_total"]  / pv["n_actions"] * 90).round(4)

    pv.to_parquet(out, index=False)
    logger.success(f"✓ VAEP para {len(pv)} jugadores")
    return pv


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    df = download_and_convert()
    if df.empty: exit(1)

    # Mostrar columnas disponibles para debug
    logger.info(f"Columnas SPADL: {list(df.columns)}")

    X, y = compute_features_labels(df)
    if X is None: exit(1)

    ms, mc = train_models(X, y)
    pv     = compute_player_vaep(df, ms, mc)

    if not pv.empty:
        nc  = "player_name" if "player_name" in pv.columns else pv.columns[0]
        top = pv.nlargest(10, "vaep_per90")[
            [nc, "vaep_per90", "offensive_per90", "defensive_per90", "n_actions"]
        ]
        print("\n🏆 Top 10 VAEP/90 — La Liga 2020/21:")
        print(top.to_string(index=False))

    print(f"\n✅ Listo. Siguiente: python vaep_apply.py")
