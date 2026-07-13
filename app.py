import os
import json
import unicodedata
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, jsonify, request
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

app = Flask(__name__)

# ── Cargar datos ──────────────────────────────────────────

def load_master():
    # Eliminamos el try-except silencioso para ver si el archivo realmente falta
    BASE_DIR = Path(__file__).resolve().parent
    csv_path = BASE_DIR / "master_players_final.csv"

    if not csv_path.exists():
        print(f"❌ ¡ALERTA! El archivo no se encuentra en: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # 1. Forzamos limpiar cualquier espacio en blanco o caracteres raros en los títulos
    df.columns = df.columns.str.strip().str.lower()

    if "minutes" in df.columns:
        df["minutes"] = df["minutes"].astype(str).str.replace(",","").str.strip()
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
    if "age" in df.columns:
        df["age_num"] = df["age"].astype(str).str.split("-").str[0]
        df["age_num"] = pd.to_numeric(df["age_num"], errors="coerce")

    num_cols = [
        "goals","assists","minutes","matches_played","yellow_cards","red_cards",
        "market_value_eur","age_num","sh","sot","sot%","g/sh","g/sot",
        "fls","fld","int","tklw","crs","ga","ga90","sota","saves","save%",
        "w","d","l","cs","cs%","pksv",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "position" in df.columns:
        df["pos_group"] = df["position"].astype(str).apply(classify_pos)

    # Añadimos un print para confirmar en la terminal que se cargó bien
    print("✅ CSV cargado con éxito. Columnas detectadas:", df.columns.tolist())
    return df

def load_vaep():
    BASE_DIR = Path(__file__).resolve().parent
    path = BASE_DIR / "data/vaep/player_vaep_argentina.csv"
    if not path.exists():
        return pd.DataFrame()
    dv = pd.read_csv(path)
    for c in ["vaep_per90","offensive_per90","defensive_per90"]:
        if c in dv.columns:
            dv[c] = pd.to_numeric(dv[c], errors="coerce")
    return dv


def load_clusters():
    BASE_DIR = Path(__file__).resolve().parent
    path = BASE_DIR / "data/ml/player_clusters.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def classify_pos(pos):
    p = str(pos).upper()
    if "GK" in p: return "Portero"
    if "DF" in p: return "Defensor"
    if "MF" in p: return "Mediocampista"
    if "FW" in p: return "Delantero"
    return "Otro"


def fmt_mv(val):
    if pd.isna(val) or val == 0: return "N/D"
    if val >= 1_000_000: return f"€{val/1_000_000:.1f}M"
    if val >= 1_000:     return f"€{val/1_000:.0f}K"
    return f"€{val:.0f}"


def norm_name(s):
    if not isinstance(s, str): return ""
    s = s.lower().strip()
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


# Cargar al iniciar
DF      = load_master()
DF_VAEP = load_vaep()
DF_CLUS = load_clusters()
MODEL_DIR = Path("data/ml/models")
if not DF_CLUS.empty:
    DF_CLUS["_norm"] = DF_CLUS["player"].apply(norm_name)

ZONA_A = [
    "Boca Juniors", "Boca", "Independiente", "San Lorenzo", "Racing Club", "Racing",
    "Vélez Sársfield", "Vélez", "Velez", "Estudiantes LP", "Estudiantes de La Plata",
    "Lanús", "Lanus", "Platense", "Newell's Old Boys", "Newell's", "Newells",
    "Rosario Central", "Central Córdoba", "Central Cordoba", "Defensa y Justicia",
    "Defensa", "Gimnasia LP", "Gimnasia (LP)", "Talleres", "Instituto",
    "Unión", "Union", "Deportivo Riestra", "Riestra", "Gimnasia (Mza)", "Gimnasia (M)"
]

ZONA_B = [
    "River Plate", "River", "Racing Club", "Racing", "Huracán", "Huracan",
    "Barracas Central", "Barracas", "Belgrano", "Estudiantes (RC)", "Estudiantes RC",
    "Argentinos Juniors", "Argentinos", "Arg Juniors", "Tigre",
    "Independiente Rivadavia", "Ind. Rivadavia", "Rosario Central", "Banfield",
    "Aldosivi", "Atlético Tucumán", "Atl Tucuman", "Atlé Tucumán", "Sarmiento"
]

FEATURES = {
    "Portero":       ["minutes","GA90","Save%","CS","CS%","PKsv","SoTA"],
    "Defensor":      ["minutes","Int","TklW","Fls","Crs","goals","assists","market_value_eur","age_num"],
    "Mediocampista": ["minutes","goals","assists","TklW","Int","Fld","Crs","market_value_eur","age_num"],
    "Delantero":     ["minutes","goals","assists","Sh","SoT","SoT%","G/Sh","G/SoT","market_value_eur","age_num"],
}


# ══════════════════════════════════════════════════════════
# RUTAS HTML
# ══════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("scout.html")

@app.route("/scout")
def scout():
    return render_template("scout.html")

@app.route("/ml")
def ml_page():
    return render_template("ml.html")

@app.route("/vaep")
def vaep_page():
    return render_template("vaep.html")

@app.route("/liga")
def liga_page():
    return render_template("liga.html")


# ══════════════════════════════════════════════════════════
# API — SCOUT
# ══════════════════════════════════════════════════════════

@app.route("/api/teams")
def api_teams():
    teams = sorted(DF["team"].dropna().unique().tolist())
    return jsonify(teams)


@app.route("/api/scout/<team>")
def api_scout(team):
    df = DF
    team_df = df[df["team"] == team]
    if team_df.empty:
        return jsonify({"error": "Equipo no encontrado"}), 404

    # Métricas generales
    metrics = {
        "jugadores":   int(len(team_df)),
        "goles":       int(team_df["goals"].sum()),
        "asistencias": int(team_df["assists"].sum()),
        "amarillas":   int(team_df["yellow_cards"].sum()),
        "valor_total": fmt_mv(team_df["market_value_eur"].sum()),
    }

    # Diagnóstico
    ideal = {"Portero":2,"Defensor":5,"Mediocampista":5,"Delantero":3}
    diagnosis = {}
    for pos, min_n in ideal.items():
        pp    = team_df[team_df["pos_group"] == pos]
        count = int(len(pp))
        weak  = bool(count < min_n or (not pp.empty and pp["minutes"].mean() < 400))
        diagnosis[pos] = {
            "count":    count,
            "ideal":    min_n,
            "weak":     weak,
            "deficit":  max(0, min_n - count),
            "avg_min":  round(pp["minutes"].mean(), 0) if not pp.empty else 0,
        }

    # Resultados (desde porteros)
    gk_df = team_df[team_df["pos_group"] == "Portero"]
    results = {}
    if not gk_df.empty:
        W = int(gk_df["w"].sum()); D = int(gk_df["d"].sum())
        L = int(gk_df["l"].sum()); CS = int(gk_df["cs"].sum())
        GA = int(gk_df["ga"].sum())
        pts = W*3 + D
        results = {"W":W,"D":D,"L":L,"CS":CS,"GA":GA,
                   "PJ":W+D+L,"Pts":pts,
                   "avg": round(pts/(W+D+L),2) if (W+D+L)>0 else 0}

    # Plantel
    cols = ["player","team","pos_group","age_num","goals","assists","minutes",
            "yellow_cards","red_cards","market_value_eur",
            "ga90","save%","cs","w","d","l","cs%","pksv","sota",
            "int","tklw","fls","crs","fld","sh","sot","sot%","g/sh","g/sot"]
    cols = [c for c in cols if c in team_df.columns]
    plantel = team_df[cols].copy()
    plantel["market_value_eur"] = plantel["market_value_eur"].apply(fmt_mv)
    plantel = plantel.sort_values("goals", ascending=False)
    plantel = plantel.fillna(0)

    COL_DISPLAY = {
        "ga90":"GA90","save%":"Save%","cs":"CS","cs%":"CS%","pksv":"PKsv","sota":"SoTA",
        "int":"Int","tklw":"TklW","fls":"Fls","crs":"Crs","fld":"Fld",
        "sh":"Sh","sot":"SoT","sot%":"SoT%","g/sh":"G/Sh","g/sot":"G/SoT",
        "w":"W","d":"D","l":"L",
    }
    records = []
    for _, r in plantel.iterrows():
        rec = {}
        for c in cols:
            rec[COL_DISPLAY.get(c, c)] = r[c]
        records.append(rec)

    return jsonify({
        "metrics":   metrics,
        "diagnosis": diagnosis,
        "results":   results,
        "plantel":   records,
    })


@app.route("/api/candidates")
def api_candidates():
    team    = request.args.get("team","")
    pos     = request.args.get("pos","Delantero")
    max_age = float(request.args.get("max_age", 30))
    max_mv  = float(request.args.get("max_mv", 10_000_000))
    min_min = float(request.args.get("min_min", 200))

    df = DF
    team_df = df[df["team"] == team]
    team_pos = team_df[team_df["pos_group"] == pos]

    pool = df[
        (df["team"] != team) &
        (df["pos_group"] == pos) &
        (df["age_num"] <= max_age) &
        (df["minutes"] >= min_min) &
        ((df["market_value_eur"] <= max_mv) | (df["market_value_eur"] == 0))
    ].copy()

    ref_name = None
    if not team_pos.empty and not pool.empty:
        if pos == "Delantero":       ref_row = team_pos.loc[team_pos["goals"].idxmax()]
        elif pos == "Mediocampista": ref_row = team_pos.loc[(team_pos["goals"]+team_pos["assists"]).idxmax()]
        else:                        ref_row = team_pos.loc[team_pos["minutes"].idxmax()]
        ref_name = ref_row["player"]

        feats = [f.lower() for f in FEATURES.get(pos,[]) if f.lower() in pool.columns]
        if feats and len(pool) > 1:
            pool_vec = pool[feats].fillna(0)
            scaler = StandardScaler()
            pool_norm = scaler.fit_transform(pool_vec)
            ref_norm  = scaler.transform(pd.DataFrame([ref_row[feats].fillna(0)]))

            CAND_WEIGHTS = {
                "Portero":       {"save%": 5.0, "sota": 4.0, "ga90": 3.0, "pksv": 3.0, "cs": 2.0, "cs%": 2.0},
                "Defensor":      {"int": 6.0, "tklw": 6.0, "crs": 3.0, "goals": 3.0, "assists": 3.0},
                "Mediocampista": {"tklw": 6.0, "int": 4.0, "crs": 3.0, "assists": 3.0, "goals": 3.0, "fld": 2.0},
                "Delantero":     {"goals": 6.0, "sh": 4.0, "sot": 4.0, "assists": 3.0, "g/sh": 2.0, "g/sot": 2.0},
            }
            w = np.array([CAND_WEIGHTS.get(pos, {}).get(f, 0.05) for f in feats])
            ref_norm  *= w
            pool_norm *= w

            distances = euclidean_distances(ref_norm, pool_norm)[0]
            median_d = float(np.median(distances))
            gamma = 1.386 / median_d if median_d > 1e-10 else 1.0
            pool["similitud"] = (np.exp(-gamma * distances) * 100).round(1)
            pool = pool.nlargest(6, "similitud")
    else:
        if pos == "Portero":         pool["_s"] = pool.get("save%",0) + pool.get("cs",0)*5
        elif pos == "Defensor":      pool["_s"] = pool.get("int",0)*3 + pool.get("tklw",0)*3
        elif pos == "Mediocampista": pool["_s"] = pool.get("goals",0)*10 + pool.get("assists",0)*8
        else:                        pool["_s"] = pool.get("goals",0)*15 + pool.get("sh",0)*0.5
        pool = pool.nlargest(6, "_s")
        pool["similitud"] = 0

    POS_STATS = {
        "Portero":       ["ga90","save%","cs","cs%","pksv","sota"],
        "Defensor":      ["int","tklw","fls","crs","goals","assists"],
        "Mediocampista": ["tklw","int","fld","crs","goals","assists"],
        "Delantero":     ["sh","sot","sot%","g/sh","g/sot","goals","assists"],
    }
    extra_cols = [c for c in POS_STATS.get(pos,[]) if c in pool.columns]
    base_cols = ["player","team","pos_group","age_num","minutes","market_value_eur","similitud"]
    all_cols = list(dict.fromkeys(base_cols + extra_cols))
    all_cols = [c for c in all_cols if c in pool.columns]
    pool_sub = pool[all_cols].fillna(0)

    COL_DISPLAY = {
        "ga90":"GA90","save%":"Save%","cs":"CS","cs%":"CS%","pksv":"PKsv","sota":"SoTA",
        "int":"Int","tklw":"TklW","fls":"Fls","crs":"Crs","fld":"Fld",
        "sh":"Sh","sot":"SoT","sot%":"SoT%","g/sh":"G/Sh","g/sot":"G/SoT",
    }
    records = []
    for _, r in pool_sub.iterrows():
        rec = {}
        for c in all_cols:
            rec[COL_DISPLAY.get(c, c)] = None if pd.isna(r[c]) else (
                float(r[c]) if isinstance(r[c], (np.integer, np.floating)) else r[c]
            )
        rec["market_value_eur_fmt"] = fmt_mv(r.get("market_value_eur", 0))
        records.append(rec)

    return jsonify({
        "ref_player": ref_name,
        "candidates": records,
    })


@app.route("/api/compare")
def api_compare():
    players_param = request.args.get("players", "")
    player_names = [p.strip() for p in players_param.split(",") if p.strip()]
    if len(player_names) < 2:
        return jsonify({"error": "Se necesitan al menos 2 jugadores"}), 400

    df = DF
    rows = []
    for name in player_names:
        r = df[df["player"] == name]
        if not r.empty:
            rows.append(r.iloc[0])
    if len(rows) < 2:
        return jsonify({"error": "Jugadores no encontrados"}), 404

    pos = rows[0].get("pos_group", "Delantero")
    feats = [f for f in FEATURES.get(pos,[]) if f.lower() in df.columns]
    if not feats:
        feats = [c for c in ["goals","assists","minutes","Sh","SoT","Int","TklW","Fld","Crs"] if c.lower() in df.columns]

    lc_feats = [f.lower() for f in feats]
    league_max = df[df["pos_group"]==pos][lc_feats].fillna(0).max().replace(0,1)
    pool = df[df["pos_group"]==pos][lc_feats].fillna(0)

    COL_KEY = {"G/Sh":"g/sh", "G/SoT":"g/sot", "SoT%":"sot%", "CS%":"cs%", "Save%":"save%", "GA90":"ga90", "PKsv":"pksv", "SoTA":"sota"}
    def _col(k):
        return COL_KEY.get(k, k.lower())

    def row_to_dict(r):
        d = {}
        for k in ["player","team","pos_group","age_num","goals","assists",
                  "minutes","market_value_eur","Sh","SoT","G/Sh","Int","TklW",
                  "GA90","Save%","CS","Fld","Crs"]:
            v = r.get(_col(k), 0)
            d[k] = float(v) if isinstance(v, (int,float,np.integer,np.floating)) else str(v)
        d["market_value_eur_fmt"] = fmt_mv(r.get("market_value_eur",0))
        return d

    radar = {
        "labels": feats,
        "players": [
            {
                "player": r.get("player",""),
                "norm": [round(float(r.get(_col(f),0) or 0) / float(league_max[_col(f)]), 3) for f in feats],
                "raw": [float(r.get(_col(f),0) or 0) for f in feats],
                "pct": [round(((pool[_col(f)] <= float(r.get(_col(f),0) or 0)).mean()*100), 1) for f in feats],
            }
            for r in rows
        ]
    }

    return jsonify({
        "players": [row_to_dict(r) for r in rows],
        "radar":   radar,
    })


@app.route("/api/players")
def api_players():
    pos = request.args.get("pos","")
    df = DF
    if pos:
        df = df[df["pos_group"] == pos]
    players = sorted(df["player"].dropna().unique().tolist())
    return jsonify(players)


# ══════════════════════════════════════════════════════════
# API — BUSCADOR ML
# ══════════════════════════════════════════════════════════

@app.route("/api/ml/positions")
def api_ml_positions():
    if DF_CLUS.empty:
        return jsonify([])
    return jsonify(["Portero","Defensor","Mediocampista","Delantero"])


@app.route("/api/ml/clusters/<pos>")
def api_ml_clusters(pos):
    if DF_CLUS.empty:
        return jsonify([])
    pos_data = DF_CLUS[DF_CLUS["pos_group"] == pos]

    AXIS_COLS = {
        "Portero":       ("ga90", "save%"),      # 💡 En minúsculas como tu limpieza de CSV
        "Defensor":      ("minutes", "int"),
        "Mediocampista": ("minutes", "fld"),
        "Delantero":     ("minutes", "sot"),
    }
    AXIS_LABELS = {
        "Portero":       ("Goles recibidos/90", "% Atajadas"),
        "Defensor":      ("Minutos jugados", "Intercepciones"),
        "Mediocampista": ("Minutos jugados", "Faltas recibidas"),
        "Delantero":     ("Minutos jugados", "Tiros al arco"),
    }

    x_col, y_col   = AXIS_COLS.get(pos, ("pca_x","pca_y"))
    x_lbl, y_lbl   = AXIS_LABELS.get(pos, ("Componente 1","Componente 2"))

    # Merge con datos reales
    plot = pos_data[["player","team","cluster_kmeans","cluster_label","pca_x","pca_y"]].copy()
    if x_col in DF.columns:
        vals = DF[["player","team",x_col,y_col]].copy()
        plot = plot.merge(vals, on=["player","team"], how="left")
        plot["x_val"] = pd.to_numeric(plot[x_col], errors="coerce").fillna(0)
        plot["y_val"] = pd.to_numeric(plot[y_col], errors="coerce").fillna(0)
    else:
        plot["x_val"] = plot["pca_x"]
        plot["y_val"] = plot["pca_y"]
    PALETA_COLORES = [
            "#00f2fe",  # Celeste / Turquesa brillante
            "#f35588",  # Rosa / Neón
            "#05dfd7",  # Verde menta
            "#fff717",  # Amarillo neón
            "#a370f7"   # Morado brillante
        ]

    clusters = []
    for idx, (lbl, grp) in enumerate(plot.groupby("cluster_label")):
            color_asignado = PALETA_COLORES[idx % len(PALETA_COLORES)]

            clusters.append({
                "label":   lbl,
                "id":      int(grp["cluster_kmeans"].iloc[0]),
                "color":   color_asignado,  # 🌟 Enviamos el color listo al frontend
                "players": grp[["player","team","x_val","y_val","cluster_kmeans","cluster_label"]].fillna(0).to_dict(orient="records"),
            })

    var_explained = None
    try:
        pca = joblib.load(MODEL_DIR / f"pca_{pos}.pkl")
        var_explained = round(pca.explained_variance_ratio_.sum() * 100, 1)
    except Exception:
        pass

    return jsonify({
        "clusters":      clusters,
        "x_label":       x_lbl,
        "y_label":       y_lbl,
        "labels":        sorted(pos_data["cluster_label"].unique().tolist()),
        "var_explained": var_explained,
    })


@app.route("/api/ml/players/<pos>")
def api_ml_players(pos):
    if DF_CLUS.empty:
        return jsonify([])
    players = sorted(DF_CLUS[DF_CLUS["pos_group"] == pos]["player"].dropna().unique().tolist())
    return jsonify(players)


@app.route("/api/ml/cluster-players/<pos>/<cluster_label>")
def api_ml_cluster_players(pos, cluster_label):
    if DF_CLUS.empty:
        return jsonify([])

    members = DF_CLUS[
        (DF_CLUS["pos_group"] == pos) &
        (DF_CLUS["cluster_label"] == cluster_label)
    ]["player"].tolist()

    players_df = DF[DF["player"].isin(members)].copy()
    cols = ["player","team","age_num","goals","assists","minutes","market_value_eur"]
    cols = [c for c in cols if c in players_df.columns]
    players_df = players_df[cols].fillna(0)
    players_df["market_value_eur_fmt"] = players_df["market_value_eur"].apply(fmt_mv)

    return jsonify(players_df.sort_values("goals", ascending=False).to_dict(orient="records"))


@app.route("/api/ml/similar")
def api_ml_similar():
    player   = request.args.get("player","")
    n        = int(request.args.get("n", 6))
    excl_team = request.args.get("excl_team","")

    if DF_CLUS.empty:
        return jsonify([])

    row = DF_CLUS[DF_CLUS["player"] == player]
    if row.empty:
        row = DF_CLUS[DF_CLUS["_norm"] == norm_name(player)]
    if row.empty:
        return jsonify([])

    r = row.iloc[0]
    pos     = r["pos_group"]
    cluster = r["cluster_kmeans"]
    actual_name = r["player"]

    try:
        scaler = joblib.load(MODEL_DIR / f"scaler_{pos}.pkl")
        feats  = joblib.load(MODEL_DIR / f"feats_{pos}.pkl")
    except Exception:
        return jsonify([])

    pool = DF_CLUS[
        (DF_CLUS["pos_group"] == pos) &
        (DF_CLUS["cluster_kmeans"] == cluster) &
        (DF_CLUS["player"] != actual_name)
    ].copy()

    if excl_team:
        pool = pool[pool["team"] != excl_team]

    if pool.empty:
        # Fallback: sin filtro de cluster, mismo pool completo de la posición
        pool = DF_CLUS[
            (DF_CLUS["pos_group"] == pos) &
            (DF_CLUS["player"] != actual_name)
        ].copy()
        if excl_team:
            pool = pool[pool["team"] != excl_team]

    if pool.empty:
        # Último recurso: mismo pool sin filtro de equipo
        pool = DF_CLUS[
            (DF_CLUS["pos_group"] == pos) &
            (DF_CLUS["player"] != actual_name)
        ].copy()

    if pool.empty:
        return jsonify([])

    ref_stats  = np.array([r[f] if pd.notna(r.get(f)) else 0 for f in feats])
    pool_stats = pool[feats].fillna(0).values
    ref_scaled  = scaler.transform(ref_stats.reshape(1, -1))
    pool_scaled = scaler.transform(pool_stats)

    POS_WEIGHTS = {
        "Portero":       {"Save%": 5.0, "SoTA": 4.0, "GA90": 3.0, "PKsv": 3.0, "CS": 2.0, "CS%": 2.0},
        "Defensor":      {"Int": 6.0, "TklW": 6.0, "Crs": 3.0, "goals": 3.0, "assists": 3.0},
        "Mediocampista": {"TklW": 6.0, "Int": 4.0, "Crs": 3.0, "assists": 3.0, "goals": 3.0, "Fld": 2.0},
        "Delantero":     {"goals": 6.0, "Sh": 4.0, "SoT": 4.0, "assists": 3.0, "G/Sh": 2.0, "G/SoT": 2.0},
    }
    for feat in feats:
        w = POS_WEIGHTS.get(pos, {}).get(feat, 0.05)
        idx = feats.index(feat)
        ref_scaled[:, idx]  *= w
        pool_scaled[:, idx] *= w

    distances = euclidean_distances(ref_scaled, pool_scaled)[0]
    pool["dist"] = distances
    median_d = float(np.median(distances))
    gamma = 1.386 / median_d if median_d > 1e-10 else 1.0
    pool["similitud_ml"] = (np.exp(-gamma * distances) * 100).round(1)
    pool["mismo_cluster"] = (pool["cluster_kmeans"] == cluster)
    pool = pool.nsmallest(n, "dist")

    result = DF[DF["player"].isin(pool["player"])].copy()
    result = result.merge(
        pool[["player","similitud_ml","cluster_label","mismo_cluster"]],
        on="player", how="left"
    )
    base_cols = ["player","team","pos_group","age_num","goals","assists",
                 "minutes","market_value_eur","similitud_ml","cluster_label","mismo_cluster"]
    STAT_COLS = {
        "Portero":       ["ga90","save%","cs","cs%","pksv","sota"],
        "Defensor":      ["int","tklw","fls","crs"],
        "Mediocampista": ["tklw","int","fld","crs"],
        "Delantero":     ["sh","sot","sot%","g/sh","g/sot"],
    }
    extra_cols = [c for c in STAT_COLS.get(pos, []) if c in result.columns]
    all_cols = list(dict.fromkeys(base_cols + extra_cols))
    all_cols = [c for c in all_cols if c in result.columns]
    result = result[all_cols].fillna(0)

    COL_DISPLAY = {
        "ga90":"GA90","save%":"Save%","cs":"CS","cs%":"CS%","pksv":"PKsv","sota":"SoTA",
        "int":"Int","tklw":"TklW","fls":"Fls","crs":"Crs","fld":"Fld",
        "sh":"Sh","sot":"SoT","sot%":"SoT%","g/sh":"G/Sh","g/sot":"G/SoT",
    }
    records = []
    for _, r in result.iterrows():
        rec = {}
        for c in all_cols:
            rec[COL_DISPLAY.get(c, c)] = None if pd.isna(r[c]) else (
                float(r[c]) if isinstance(r[c], (np.integer, np.floating)) else r[c]
            )
        rec["market_value_eur_fmt"] = fmt_mv(r.get("market_value_eur", 0))
        records.append(rec)

    return jsonify(records)


@app.route("/api/ml/player-info")
def api_ml_player_info():
    player = request.args.get("player", "")
    if DF_CLUS.empty or not player:
        return jsonify({})
    row = DF_CLUS[DF_CLUS["player"] == player]
    if row.empty:
        row = DF_CLUS[DF_CLUS["_norm"] == norm_name(player)]
    if row.empty:
        return jsonify({})
    r = row.iloc[0]
    return jsonify({
        "player":        r["player"],
        "team":          r["team"],
        "pos_group":     r["pos_group"],
        "cluster_kmeans":int(r["cluster_kmeans"]),
        "cluster_label": r["cluster_label"],
        "cluster_dbscan":int(r["cluster_dbscan"]),
        "pca_x":         float(r["pca_x"]),
        "pca_y":         float(r["pca_y"]),
    })


# ══════════════════════════════════════════════════════════
# API — VAEP
# ══════════════════════════════════════════════════════════

@app.route("/api/vaep")
def api_vaep():
    dv = DF_VAEP.copy()
    if dv.empty:
        return jsonify([])

    team = request.args.get("team","")
    pos  = request.args.get("pos","")
    min_m = float(request.args.get("min_min",200))

    if "minutes" in dv.columns:
        dv["minutes"] = pd.to_numeric(dv["minutes"], errors="coerce").fillna(0)
        dv = dv[dv["minutes"] >= min_m]
    if team: dv = dv[dv["team"] == team]
    if pos:  dv = dv[dv["pos_group"] == pos]

    cols = ["player","team","pos_group","goals","assists","vaep_per90","offensive_per90","defensive_per90"]
    cols = [c for c in cols if c in dv.columns]
    dv = dv[cols].fillna(0)
    return jsonify(dv.sort_values("vaep_per90", ascending=False).to_dict(orient="records"))


# ══════════════════════════════════════════════════════════
# API — LIGA
# ══════════════════════════════════════════════════════════

@app.route("/api/liga/stats")
def api_liga_stats():
    df = DF
    return jsonify({
        "jugadores": int(len(df)),
        "equipos":   int(df["team"].nunique()),
        "goles":     int(df["goals"].sum()),
        "valor_prom":fmt_mv(df["market_value_eur"].mean()),
    })

@app.route("/api/liga/standings")
def api_liga_standings():
    df = DF
    gk_df = df[df["pos_group"] == "Portero"].copy()
    if gk_df.empty:
        return jsonify({"zona_a": [], "zona_b": []})

    # 1. Agrupamos estadísticas reales por equipo desde los porteros
    gk_team = gk_df.groupby("team").agg(
        G  = ("w", "sum"),
        E  = ("d", "sum"),
        P  = ("l", "sum"),
        GC = ("ga", "sum"),
        VI = ("cs", "sum"),
    ).reset_index()

    # Convertimos los valores a enteros nativos de Python para evitar errores de JSON
    gk_team["G"] = gk_team["G"].astype(int)
    gk_team["E"] = gk_team["E"].astype(int)
    gk_team["P"] = gk_team["P"].astype(int)
    gk_team["GC"] = gk_team["GC"].astype(int)
    gk_team["VI"] = gk_team["VI"].astype(int)

    # PJ y Puntos reales calculados
    gk_team["PJ"] = gk_team["G"] + gk_team["E"] + gk_team["P"]
    gk_team["Pts"] = (gk_team["G"] * 3) + gk_team["E"]

    # Goles a Favor (GF) extraídos de la suma de goles de sus jugadores
    gk_team["GF"] = gk_team["team"].apply(lambda t: int(df[df["team"] == t]["goals"].sum()))
    gk_team["DG"] = gk_team["GF"] - gk_team["GC"]

    # 2. LISTAS DE MAPEO REAL DE LA LIGA PROFESIONAL (Nombres exactos del CSV)
    equipos_zona_a = [
            "Boca Juniors",
            "Unión",
            "Gimnasia–M",
            'Vélez Sarsfield',
            'Talleres–C',
            "C. Córdoba–SdE",     # Ajustado al CSV
            "Defensa",
            "Lanús",
            "Newell's",
            "Estudiantes–LP",
            "Instituto",
            "Independiente",
            "Platense",
            "Dep. Riestra",       # Ajustado al CSV
            "CA San Lorenzo",     # Ajustado al CSV
        ]

    # Los demás irán a la Zona B automáticamente (donde pertenecen Aldosivi, River, Racing, etc.)
    def clasificar_zona(team_name):
        # Si coincide con alguno de la lista A, va a la A; de lo contrario va a la B
        if any(eq.lower() in team_name.lower() for eq in equipos_zona_a):
            return "A"
        return "B"

    gk_team["zona"] = gk_team["team"].apply(clasificar_zona)

    # 3. Separar los grupos
    zona_a_df = gk_team[gk_team["zona"] == "A"].copy()
    zona_b_df = gk_team[gk_team["zona"] == "B"].copy()

    # 4. Ordenar competitivamente por Pts -> DG -> GF
    zona_a_df = zona_a_df.sort_values(by=["Pts", "DG", "GF"], ascending=False)
    zona_b_df = zona_b_df.sort_values(by=["Pts", "DG", "GF"], ascending=False)

    return jsonify({
        "zona_a": zona_a_df.to_dict(orient="records"),
        "zona_b": zona_b_df.to_dict(orient="records")
    })

@app.route("/api/liga/top-scorers")
def api_liga_top_scorers():
    df = DF.nlargest(15,"goals")[["player","team","pos_group","goals","assists"]].fillna(0)
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/liga/team-values")
def api_liga_team_values():
    df = DF.groupby("team")["market_value_eur"].sum().nlargest(15).reset_index()
    df.columns = ["team","value"]
    df["value_fmt"] = df["value"].apply(fmt_mv)
    return jsonify(df.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
