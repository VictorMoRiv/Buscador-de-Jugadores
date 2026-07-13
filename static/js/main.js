"use strict";

// ── Paleta de colores dinámica ─────────────────────────
let C = {};
let _prevC = {};

function replaceChartColor(str) {
  if (typeof str !== "string") return str;
  for (const k of Object.keys(_prevC)) {
    if (str.startsWith(_prevC[k])) return str.replace(_prevC[k], C[k]);
  }
  return str;
}

function refreshChartColors() {
  Object.values(state.charts).forEach(ch => {
    if (!ch) return;
    ch.data.datasets.forEach(ds => {
      if (Array.isArray(ds.backgroundColor)) {
        ds.backgroundColor = ds.backgroundColor.map(replaceChartColor);
      } else if (typeof ds.backgroundColor === "string") {
        ds.backgroundColor = replaceChartColor(ds.backgroundColor);
      }
      if (typeof ds.borderColor === "string") ds.borderColor = replaceChartColor(ds.borderColor);
      if (typeof ds.pointBackgroundColor === "string") ds.pointBackgroundColor = replaceChartColor(ds.pointBackgroundColor);
    });
    if (ch.options.scales) {
      Object.values(ch.options.scales).forEach(s => {
        if (s.ticks) s.ticks.color = C.text2;
        if (s.grid) s.grid.color = C.border;
        if (s.pointLabels) s.pointLabels.color = C.text2;
        if (s.title) s.title.color = C.text3;
      });
    }
    if (ch.options.plugins?.legend?.labels) ch.options.plugins.legend.labels.color = C.text;
    ch.update();
  });
}

function updateJSColors(isDark) {
  _prevC = {...C};
  if (isDark) {
    C.surface  = "#282828";
    C.surface2 = "#3c3836";
    C.border   = "#504945";
    C.text     = "#fbf1c7";
    C.text2    = "#bdae93";
    C.text3    = "#665c54";
    C.blue     = "#83a598";
    C.gold     = "#fabd2f";
    C.green    = "#b8bb26";
    C.red      = "#fb4934";
    C.purple   = "#d3869b";
  } else {
    C.surface  = "#faf4ed";
    C.surface2 = "#fffaf3";
    C.border   = "#f2e9e1";
    C.text     = "#575279";
    C.text2    = "#797593";
    C.text3    = "#9893a5";
    C.blue     = "#56949f";
    C.gold     = "#ea9d34";
    C.green    = "#286983";
    C.red      = "#b4637a";
    C.purple   = "#907aa9";
  }

  if (window.Chart) {
    Chart.defaults.color = C.text2;
    Chart.defaults.borderColor = C.border;
  }
  refreshChartColors();
}

function posColor(pos) {
  const map = { Portero: C.blue, Defensor: C.green, Mediocampista: C.gold, Delantero: C.red };
  return map[pos] || C.blue;
}

const POS_BADGE_CLASS = {
  Portero: "pos-GK", Defensor: "pos-DF",
  Mediocampista: "pos-MF", Delantero: "pos-FW",
};
const POS_BADGE_LABEL = {
  Portero: "GK", Defensor: "DF",
  Mediocampista: "MF", Delantero: "FW",
};

function getClusterColors() {
  return [C.blue, C.red, C.green, C.gold, C.purple, "#2AC3DE"];
}

function getTeamShield(teamName) {
  if (!teamName) return "";
  const cleanName = teamName.toLowerCase().trim();
  return `/static/escudos/${cleanName}.png`;
}

Chart.defaults.color = C.text2;
Chart.defaults.borderColor = C.border;
Chart.defaults.font.family = "'DM Mono', monospace";

// ── State ─────────────────────────────────────
const state = {
  currentTeam:    "",
  currentPos:     "Delantero",
  mlPos:          "Portero",
  mlCluster:      "",
  charts:         {},
  plantelFull:    [],
  vaepData:       [],
  compareChart:   "radar",
};

// ── Utils ─────────────────────────────────────
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];
const api = (url) => fetch(url).then(r => r.json());

function posBadge(pos) {
  const cls = POS_BADGE_CLASS[pos] || "pos-DF";
  const lbl = POS_BADGE_LABEL[pos] || pos?.slice(0,2) || "?";
  return `<span class="pos-badge ${cls}">${lbl}</span>`;
}

function simBar(pct) {
  return `<div class="sim-bar">
    <div class="sim-label">Similitud <span>${Math.round(pct)}%</span></div>
    <div class="sim-bg"><div class="sim-fill" style="width:${pct}%"></div></div>
  </div>`;
}

function playerCard(p, opts = {}) {
  const { showSim = false, gold = false, clusterTag = "" } = opts;
  const pos    = p.pos_group || "";
  const age    = p.age_num > 0 ? `${Math.round(p.age_num)} años` : "";
  const mvFmt  = p.market_value_eur_fmt || "N/D";

  // Stats según posición
  const statDefs = {
    Portero:       [["GA90","GA/90"],["Save%","Ataj.%"],["CS","V.Inv."],["W","Vict."]],
    Defensor:      [["goals","Goles"],["assists","Asist."],["Int","Interc."],["TklW","Tackles"]],
    Mediocampista: [["goals","Goles"],["assists","Asist."],["TklW","Tackles"],["Fld","F.Rec."]],
    Delantero:     [["goals","Goles"],["assists","Asist."],["Sh","Tiros"],["SoT","Al arco"]],
  };
  const stats = (statDefs[pos] || statDefs.Delantero).map(([k, lbl]) => {
    const v = p[k] || 0;
    const fmt = (k === "Save%" || k === "SoT%") ? `${(+v).toFixed(1)}%`
              : (k === "GA90" || k === "G/Sh")  ? `${(+v).toFixed(2)}`
              : `${Math.round(+v)}`;
    return `<div class="pc-stat">
      <span class="pc-stat-val">${fmt}</span>
      <span class="pc-stat-lbl">${lbl}</span>
    </div>`;
  }).join("");

  return `
  <div class="player-card ${gold ? "gold" : ""}">
    ${clusterTag ? `<div class="cluster-tag ${clusterTag.same ? "same":"other"}">${clusterTag.label}</div>` : ""}
    <div class="pc-top">
      <div>
        <div class="pc-name">${p.player}</div>
        <div class="pc-team">${p.team} · ${age}</div>
      </div>
      <div style="display:flex;gap:6px;align-items:center">
        ${gold ? `<span style="color:${C.gold};font-size:10px;font-family:'DM Mono'">PLANTEL</span>` : ""}
        ${posBadge(pos)}
      </div>
    </div>
    <div class="pc-stats">${stats}
      <div class="pc-stat">
        <span class="pc-stat-val">${mvFmt}</span>
        <span class="pc-stat-lbl">Valor</span>
      </div>
    </div>
    ${showSim && p.similitud !== undefined ? simBar(p.similitud) : ""}
    ${showSim && p.similitud_ml !== undefined ? simBar(p.similitud_ml) : ""}
  </div>`;
}

function destroyChart(key) {
  if (state.charts[key]) {
    state.charts[key].destroy();
    delete state.charts[key];
  }
}

// ══════════════════════════════════════════════
// NAVEGACIÓN — no más SPA, cada ruta es su página
// ══════════════════════════════════════════════
const CURRENT_PAGE = window.location.pathname.replace(/\/$/, "") || "/scout";

// ══════════════════════════════════════════════
// SCOUT — INIT
// ══════════════════════════════════════════════
async function initScout() {
  console.log("initScout start");
  const teams = await api("/api/teams");
  console.log("initScout: teams loaded", teams.length);
  const sel = $("#team-select");

  if (!sel) return; // 👈 Evita errores si el selector aún no existe en pantalla

  sel.innerHTML = teams.map(t => `<option value="${t}">${t}</option>`).join("");
  state.currentTeam = teams[0];

  // 👈 Manejo seguro del escudo inicial
  const shieldImg = $("#team-select-shield");
  if (shieldImg) {
    shieldImg.src = getTeamShield(teams[0]);
    shieldImg.style.display = "block";
  }

  await loadScout(teams[0]);

    sel.addEventListener("change", async () => {
    state.currentTeam = sel.value;

    const currentShield = $("#team-select-shield");
    if (currentShield) {
      currentShield.src = getTeamShield(sel.value);
      currentShield.style.display = "block";
    }

      await loadScout(sel.value);
      updateCmpP1Team();
    });

  // Plantel tabs
  $$("#plantel-tabs .tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      $$("#plantel-tabs .tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderPlantel(btn.dataset.pos);
      $$("#diagnosis-body .diag-legend-item").forEach(el => {
        el.style.background = el.dataset.pos === btn.dataset.pos ? "var(--blue-dim)" : "";
      });
    });
  });

  // Filters
  setupRangeDisplay("f-age", "f-age-val", v => v);
  setupRangeDisplay("f-mv",  "f-mv-val",  v => `€${v}M`);
  setupRangeDisplay("f-min", "f-min-val", v => `${v}'`);

  // Pos toggle
  $$("#pos-toggle .pos-btn").forEach(btn => {
    btn.addEventListener("click", () => btn.classList.toggle("active"));
  });

  $("#btn-search").addEventListener("click", loadCandidates);

  // Compare players
  console.log("initScout: about to loadPlayersSelect");
  await loadPlayersSelect();
  console.log("initScout: about to updateCmpP1Team");
  updateCmpP1Team();
  console.log("initScout: setting up compare buttons");
  $("#btn-compare").addEventListener("click", loadCompare);
  $("#btn-cmp-add").addEventListener("click", addCompareSlot);

  // Chart tabs
  $$(".chart-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      $$(".chart-tab").forEach(t => t.classList.remove("active"));
      $$(".chart-canvas").forEach(c => {
        c.classList.remove("active");
        c.style.display = "none";
      });
      tab.classList.add("active");
      state.compareChart = tab.dataset.chart;
      const target = $(`#chart-${tab.dataset.chart}`);
      if (target) {
        target.classList.add("active");
        target.style.display = target.id === "chart-percentile" ? "grid" : "block";
        if (target.id !== "chart-percentile") {
          const ch = state.charts[tab.dataset.chart];
          if (ch && typeof ch.resize === "function") {
            requestAnimationFrame(() => ch.resize());
          }
        }
      }
    });
  });
}

function setupRangeDisplay(inputId, valId, fmt) {
  const input = $(`#${inputId}`);
  const val   = $(`#${valId}`);
  if (!input || !val) return;
  val.textContent = fmt(input.value);
  input.addEventListener("input", () => { val.textContent = fmt(input.value); });
}

// ══════════════════════════════════════════════
// SCOUT — CARGAR EQUIPO
// ══════════════════════════════════════════════
async function loadScout(team) {
  let data;
  try {
    data = await api(`/api/scout/${encodeURIComponent(team)}`);
  } catch {
    console.error("loadScout: error fetching data for", team);
    return;
  }
  if (data.error) {
    console.error("loadScout: API error for", team, data.error);
    return;
  }
  const infoText = $("#nombre-equipo");
  if (infoText) {
      infoText.innerHTML = `${team} tuvo estas estadísticas en la ultima temporada:`;
    }
  // Métricas
  const m = data.metrics || {};
  const mEl = Object.entries({
    "Total de Jugadores":   [m.jugadores,   null],
    "Goles en total":       [m.goles,       C.green],
    "Asistencias en total": [m.asistencias, C.blue],
    "Tarjetas Amarillas en toda la temporada":   [m.amarillas,   C.gold],
    "Valor total de la plantilla": [m.valor_total, C.purple],
  });
  $("#scout-metrics").innerHTML = mEl.map(([lbl,[val,color]]) => `
    <div class="metric-card">
      <span class="metric-val" style="color:${color || C.blue}">${val}</span>
      <span class="metric-lbl">${lbl}</span>
    </div>`).join("");

  // Diagnóstico
  renderDiagnosis(data.diagnosis);

  // Resultados
  renderResults(data.results);

  // Plantel
  state.plantelFull = data.plantel || [];
  renderPlantel("");
}

function renderDiagnosis(diag) {
  const FIELDS = {
    Portero:       { label: "POR",  tag: "Portero" },
    Defensor:      { label: "DEF", tag: "Defensa" },
    Mediocampista: { label: "MED", tag: "Mediocampo" },
    Delantero:     { label: "DEL", tag: "Ataque" },
  };
  function covClass(info) {
    if (!info.weak) return "cov-ok";
    if (info.deficit > 0) return "cov-bad";
    return "cov-warn";
  }

  // 4-3-3 rows (top to bottom = attack to GK)
  const rows = [
    { players: ["DEL","DEL","DEL"], tags: ["EI","DC","ED"] },
    { players: ["MED","MED","MED"], tags: ["MC","MC","MC"] },
    { players: ["DEF","DEF","DEF","DEF"], tags: ["LI","DFC","DFC","LD"] },
    { players: ["POR"], tags: ["PO"] },
  ];

  let pitchHtml = "";
  rows.forEach((row, ri) => {
    let rowHtml = row.players.map((pl, ci) => {
      // Find which position group this belongs to
      let posGroup = null;
      for (const [pg, f] of Object.entries(FIELDS)) {
        if (f.label === pl) { posGroup = pg; break; }
      }
      if (!posGroup) return "";
      const info = diag[posGroup];
      if (!info) return "";
      return `<div class="fp ${covClass(info)}" data-pos="${posGroup}" title="${posGroup}: ${info.count}/${info.ideal}">
        ${row.tags[ci]}
        <span class="fp-tag">${pl}</span>
      </div>`;
    }).join("");
    pitchHtml += `<div class="pitch-row${ri === 0 ? " top" : ri === rows.length-1 ? " bottom" : ""}">${rowHtml}</div>`;
  });

  // Side stats
  let sideHtml = "";
  for (const [pos, info] of Object.entries(diag)) {
    const cls = covClass(info);
    const status = info.weak
      ? (info.deficit > 0 ? `−${info.deficit}` : "Bajo")
      : "OK";
    const f = FIELDS[pos];
    sideHtml += `<div class="diag-legend-item" data-pos="${pos}">
      <span class="fp-sm ${cls}">${f.label}</span>
      <span>${f.tag}</span>
      <span class="dl-count">${info.count}/${info.ideal}</span>
      <span class="dl-status ${cls}">${status}</span>
    </div>`;
  }

  $("#diagnosis-body").innerHTML = `
    <div class="formation-wrap">
      <div class="formation-pitch">${pitchHtml}</div>
      <div class="formation-side">${sideHtml}</div>
    </div>`;

  // Clicks
  $$("#diagnosis-body .fp, #diagnosis-body .diag-legend-item").forEach(el => {
    el.addEventListener("click", () => {
      const pos = el.dataset.pos;
      if (!pos) return;
      const btn = $(`#plantel-tabs .tab-btn[data-pos="${pos}"]`);
      if (btn) {
        $$("#plantel-tabs .tab-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        renderPlantel(pos);
        $$("#diagnosis-body .diag-legend-item").forEach(li => {
          li.style.background = li.dataset.pos === pos ? "var(--blue-dim)" : "";
        });
      }
    });
  });
}

function renderResults(r) {
  if (!r || !r.PJ) {
    $("#wdl-row").innerHTML = `<span style="color:${C.text3}">Sin datos de resultados</span>`;
    return;
  }
  $("#wdl-row").innerHTML = `
    <div class="wdl-item"><div class="wdl-num wdl-w">${r.W}</div><div class="wdl-lbl">Victorias</div></div>
    <div class="wdl-item"><div class="wdl-num wdl-d">${r.D}</div><div class="wdl-lbl">Empates</div></div>
    <div class="wdl-item"><div class="wdl-num wdl-l">${r.L}</div><div class="wdl-lbl">Derrotas</div></div>
    <div class="wdl-item"><div class="wdl-num" style="color:${C.blue}">${r.CS}</div><div class="wdl-lbl">V.Inv.</div></div>
  `;
  $("#results-pts").textContent = `${r.Pts} pts · ${r.avg} x partido`;

  destroyChart("wdl");
  if (typeof Chart === "undefined") return;
  const canvas = $("#wdl-chart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  state.charts.wdl = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Victorias","Empates","Derrotas"],
      datasets: [{
        data: [r.W, r.D, r.L],
        backgroundColor: [C.green, C.text3, C.red],
        borderRadius: 4, borderSkipped: false,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: C.border }, ticks: { color: C.text2 } },
        y: { grid: { color: C.border }, ticks: { color: C.text2, stepSize: 1 } },
      }
    }
  });
}

function renderPlantel(filterPos) {
  const rows = state.plantelFull.filter(p => !filterPos || p.pos_group === filterPos);
  $("#plantel-body").innerHTML = rows.map(p => `
    <tr>
      <td class="player-name">${p.player}</td>
      <td>${posBadge(p.pos_group)}</td>
      <td class="mono">${p.age_num > 0 ? Math.round(p.age_num) : "—"}</td>
      <td class="mono">${p.goals}</td>
      <td class="mono">${p.assists}</td>
      <td class="mono">${p.minutes?.toLocaleString()}'</td>
      <td class="mono">${p.yellow_cards}</td>
      <td class="mono">${p.market_value_eur}</td>
    </tr>`).join("");
}

// ══════════════════════════════════════════════
// CANDIDATOS
// ══════════════════════════════════════════════
async function loadCandidates() {
  const team   = state.currentTeam;
  const maxAge = $("#f-age").value;
  const maxMv  = $("#f-mv").value * 1_000_000;
  const minMin = $("#f-min").value;
  const activePosbtns = $$("#pos-toggle .pos-btn.active");
  const positions = activePosbtns.map(b => b.dataset.pos);

  if (!positions.length) {
    alert("Seleccioná al menos una posición.");
    return;
  }

  const grid = $("#candidates-grid");
  grid.innerHTML = `<div class="skeleton"></div>`.repeat(6);

  const results = await Promise.all(
    positions.map(pos =>
      api(`/api/candidates?team=${encodeURIComponent(team)}&pos=${pos}&max_age=${maxAge}&max_mv=${maxMv}&min_min=${minMin}`)
    )
  );

  let html = "";
  positions.forEach((pos, i) => {
    const { ref_player, candidates } = results[i];
    if (!candidates?.length) return;
    const color = posColor(pos);
    html += `<div style="grid-column:1/-1;margin:8px 0 4px">
      <span style="font-family:'Outfit';font-weight:700;font-size:14px;color:${color}">${pos}s</span>
      ${ref_player ? `<span style="font-size:11px;color:${C.text3};margin-left:8px">similitud vs ${ref_player}</span>` : ""}
    </div>`;
    candidates.forEach(p => { html += playerCard(p, { showSim: !!ref_player }); });
  });

  grid.innerHTML = html || `<p style="color:${C.text3}">No se encontraron candidatos con esos filtros.</p>`;
}

// ══════════════════════════════════════════════
// COMPARADOR
// ══════════════════════════════════════════════
function getCmpColors() { return [C.blue, C.red, C.green, C.gold, C.purple, "#2AC3DE", "#FF6B6B", "#51CF66"]; }

function buildAc(input, dropdown, items) {
  let open = false;
  let highlightIdx = -1;

  function render(filter) {
    const q = (filter || "").toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
    const matches = items.filter(n => n.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "").includes(q));
    if (!matches.length) {
      dropdown.innerHTML = `<div class="ac-empty">Sin resultados</div>`;
      dropdown.classList.add("open");
      open = true;
      return;
    }
    dropdown.innerHTML = matches.map((n, i) =>
      `<div class="ac-item" data-idx="${i}">${n}</div>`
    ).join("");
    dropdown.classList.add("open");
    open = true;
    highlightIdx = -1;
    $$(".ac-item", dropdown).forEach(el => {
      el.addEventListener("click", () => {
        input.value = el.textContent;
        dropdown.classList.remove("open");
        open = false;
      });
    });
  }

  input.addEventListener("input", () => render(input.value));

  input.addEventListener("focus", () => {
    render(input.value);
  });

  input.addEventListener("blur", () => {
    setTimeout(() => { dropdown.classList.remove("open"); open = false; }, 200);
  });

  input.addEventListener("keydown", (e) => {
    const items = $$(".ac-item", dropdown);
    if (!open || !items.length) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      highlightIdx = Math.min(highlightIdx + 1, items.length - 1);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      highlightIdx = Math.max(highlightIdx - 1, 0);
    } else if (e.key === "Enter" && highlightIdx >= 0) {
      e.preventDefault();
      input.value = items[highlightIdx].textContent;
      dropdown.classList.remove("open"); open = false;
      return;
    } else return;
    items.forEach((el, i) => el.classList.toggle("highlight", i === highlightIdx));
    if (highlightIdx >= 0) items[highlightIdx].scrollIntoView({ block: "nearest" });
  });
}

async function loadPlayersSelect() {
  console.log("loadPlayersSelect start");
  const players = await api("/api/players");
  console.log("loadPlayersSelect: players=", players.length);

  const acAll = $("#ac-all");
  const acTeam = $("#ac-team");
  if (acAll) buildAc($(".cmp-p-extra"), acAll, players);
  if (acTeam) {
    const p1 = $("#cmp-p1");
    buildAc(p1, acTeam, []);
    state._updateTeamAc = (names) => {
      acTeam.innerHTML = "";
      buildAc(p1, acTeam, names);
    };
  }
  console.log("loadPlayersSelect: autocomplete set up");
}

function updateCmpP1Team() {
  const team = state.currentTeam;
  if (!team) { console.log("updateCmpP1Team: no team"); return; }
  const p1 = $("#cmp-p1");
  if (!p1) { console.log("updateCmpP1Team: p1 not found"); return; }
  p1.placeholder = `Jugadores de ${team}...`;
  p1.value = "";
  const allPlantel = state.plantelFull;
  if (!allPlantel || !Array.isArray(allPlantel)) {
    console.log("updateCmpP1Team: plantelFull not ready");
    return;
  }
  const names = allPlantel.map(p => p.player).filter(Boolean);
  if (state._updateTeamAc) state._updateTeamAc(names);
  console.log("updateCmpP1Team: team ac updated", names.length);
}
function addCompareSlot() {
  const container = $("#cmp-slots");

  const slot = document.createElement("div");
  slot.className = "cmp-slot";
  const uid = "ac-" + Date.now();
  slot.innerHTML = `
    <label class="select-label">OTRO JUGADOR</label>
    <div class="ac-wrap">
      <input class="select-input cmp-p-extra" placeholder="Escribí para buscar...">
      <div class="ac-dropdown" id="${uid}"></div>
    </div>
    <button class="cmp-remove" title="Quitar">×</button>`;

  const input = slot.querySelector(".cmp-p-extra");
  const dd = slot.querySelector(".ac-dropdown");
  api("/api/players").then(players => buildAc(input, dd, players));

  slot.querySelector(".cmp-remove").addEventListener("click", () => {
    slot.remove();
  });

  container.appendChild(slot);
}

async function loadCompare() {
  const inputs = [$("#cmp-p1"), ...$$(".cmp-p-extra")];
  const players = inputs.map(s => s?.value?.trim()).filter(Boolean);
  if (players.length < 2) {
    alert("Agregá al menos 2 jugadores.");
    return;
  }

  const data = await api(`/api/compare?players=${players.map(encodeURIComponent).join(",")}`);
  if (data.error) { alert(data.error); return; }

  const res = $("#compare-result");
  res.classList.remove("hidden");

  // Cards
  const colors = getCmpColors();
  $("#compare-cards").innerHTML = data.players.map((p, i) =>
    playerCard(p, { gold: p.team === state.currentTeam })
  ).join("");

  // Chart tabs — solo radar al inicio
  renderPercentiles(data);
  renderBarsCompare(data);
  renderRadar(data);
  // Asegurar que solo radar esté visible
  $$(".chart-canvas").forEach(c => { c.classList.remove("active"); c.style.display = "none"; });
  const radarCanvas = $("#chart-radar");
  if (radarCanvas) { radarCanvas.classList.add("active"); radarCanvas.style.display = "block"; }
  if (state.charts.radar && typeof state.charts.radar.resize === "function") {
    requestAnimationFrame(() => state.charts.radar.resize());
  }

  // Table headers
  const thead = $("#cmp-thead");
  thead.innerHTML = `<tr>
    <th>Estadística</th>
    ${data.players.map((p, i) =>
      `<th style="color:${colors[i % colors.length]}">${p.player.split(" ").pop()}</th>`
    ).join("")}
    <th style="color:${C.text3}">Mejor</th>
  </tr>`;

  // Table body
  const FEAT_LABELS = {
    minutes:"Minutos", goals:"Goles", assists:"Asist.",
    Int:"Intercepciones", TklW:"Tackles", Fld:"Faltas rec.",
    Fls:"Faltas com.", Crs:"Centros", Sh:"Tiros", SoT:"Al arco",
    "SoT%":"% Al arco", "G/Sh":"Conv./tiro", "G/SoT":"Conv./arco",
    GA90:"Goles rec./90", "Save%":"% Atajadas", CS:"Vallas inv.",
    "CS%":"% Vallas inv.", PKsv:"Penales ataj.", SoTA:"Tiros recib.",
    market_value_eur:"Valor mercado", age_num:"Edad",
  };
  const feats = data.radar.labels;
  const fmt = (v, f) => (f.includes("%") ? (+v).toFixed(1)+"%" : f.includes("90") ? (+v).toFixed(2) : Math.round(+v));
  
  const rows = feats.map((f, fi) => {
    const vals = data.radar.players.map(p => p.raw[fi]);
    const maxVal = Math.max(...vals);
    const bestPlayers = data.players.filter((_, i) => vals[i] === maxVal).map(p => p.player.split(" ").pop());
    return `<tr>
      <td>${FEAT_LABELS[f]||f}</td>
      ${data.radar.players.map((p, i) => {
        const isBest = vals[i] === maxVal;
        return `<td class="mono ${isBest ? "text-green" : ""}">${fmt(p.raw[fi], f)}</td>`;
      }).join("")}
      <td style="color:${C.text3};font-size:11px">${bestPlayers.join(" / ")}</td>
    </tr>`;
  });
  $("#compare-body").innerHTML = rows.join("");
}

function renderRadar(data) {
  destroyChart("radar");
  if (typeof Chart === "undefined") return;
  const canvas = $("#chart-radar");
  if (!canvas) return;
  const feats  = data.radar.labels;
  const FEAT_LABELS = {
    minutes:"Min", goals:"Goles", assists:"Asist", Int:"Interc",
    TklW:"Tackles", Fld:"F.Rec", Sh:"Tiros", SoT:"Al arco",
    "SoT%":"SoT%", "G/Sh":"Conv", GA90:"GA/90", "Save%":"Ataj%",
    CS:"V.Inv",
  };
  const colors = getCmpColors();
  const ctx = canvas.getContext("2d");
  state.charts.radar = new Chart(ctx, {
    type: "radar",
    data: {
      labels: feats.map(f => FEAT_LABELS[f]||f),
      datasets: data.radar.players.map((p, i) => ({
        label: p.player,
        data: p.norm,
        borderColor: colors[i % colors.length],
        backgroundColor: colors[i % colors.length]+"30",
        pointBackgroundColor: colors[i % colors.length],
        borderWidth: 2,
      }))
    },
    options: {
      responsive: true,
      scales: {
        r: {
          min: 0, max: 1,
          grid: { color: C.border },
          pointLabels: { color: C.text2, font: { size: 11 } },
          ticks: { display: false },
          angleLines: { color: C.border },
        }
      },
      plugins: {
        legend: { labels: { color: C.text, boxWidth: 12 } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const i = ctx.dataIndex;
              const raw = data.radar.players[ctx.datasetIndex].raw[i];
              return ` ${ctx.dataset.label}: ${typeof raw === "number" ? raw.toFixed(2) : raw}`;
            }
          }
        }
      }
    }
  });
}

function renderBarsCompare(data) {
  destroyChart("bars");
  if (typeof Chart === "undefined") return;
  const canvas = $("#chart-bars");
  if (!canvas) return;
  const FEAT_LABELS = {
    minutes:"Min", goals:"Goles", assists:"Asist", Int:"Interc",
    TklW:"Tackles", Fld:"F.Rec", Sh:"Tiros", SoT:"Al arco",
    "SoT%":"SoT%","G/Sh":"Conv","GA90":"GA/90","Save%":"Ataj%",CS:"V.Inv",
  };
  const labels = data.radar.labels.map(f => FEAT_LABELS[f]||f);
  const colors = getCmpColors();
  const ctx = canvas.getContext("2d");
  state.charts.bars = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: data.radar.players.map((p, i) => ({
        label: p.player,
        data: p.norm.map(v => +(v*100).toFixed(1)),
        backgroundColor: colors[i % colors.length]+"BB",
        borderRadius: 3,
      }))
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: C.text } } },
      scales: {
        x: { grid:{ color:C.border }, ticks:{ color:C.text2, maxRotation:35 } },
        y: {
          grid:{ color:C.border }, ticks:{ color:C.text2 },
          title:{ display:true, text:"% del máximo liga", color:C.text3 },
          max: 110,
        }
      }
    }
  });
}

function renderPercentiles(data) {
  const FEAT_LABELS = {
    minutes:"Minutos", goals:"Goles", assists:"Asist.", Int:"Interc.",
    TklW:"Tackles", Fld:"F.Rec.", Sh:"Tiros", SoT:"Al arco",
    "SoT%":"SoT%","G/Sh":"Conv.","GA90":"GA/90","Save%":"Ataj%",CS:"V.Inv.",
  };
  const colors = getCmpColors();
  const feats = data.radar.labels;
  const html = feats.map((f, fi) => {
    const pcts = data.radar.players.map(p => p.pct[fi]);
    return `
    <div class="pct-item">
      <div class="pct-label">${FEAT_LABELS[f]||f}</div>
      ${data.radar.players.map((p, i) => {
        const pct = pcts[i];
        const color = pct >= 66 ? C.green : pct >= 33 ? C.gold : C.red;
        return `
        <div class="pct-bars">
          <div class="pct-bar-wrap">
            <div class="pct-bar-fill" style="width:${pct}%;background:${color}"></div>
          </div>
          <div class="pct-val" style="color:${colors[i % colors.length]}">P${Math.round(pct)}</div>
        </div>`;
      }).join("")}
    </div>`;
  }).join("");
  $("#chart-percentile").innerHTML = html;
}

// ══════════════════════════════════════════════
// ML — BUSCADOR
// ══════════════════════════════════════════════
async function initML() {
  state.mlLoaded = true;
  setupRangeDisplay("ml-n", "ml-n-val", v => v);

  $$("#ml-pos-group .pos-radio").forEach(btn => {
    btn.addEventListener("click", () => {
      $$("#ml-pos-group .pos-radio").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.mlPos = btn.dataset.pos;
      loadMLClusters(state.mlPos);
    });
  });

  $("#ml-cluster-select").addEventListener("change", () => {
    state.mlCluster = $("#ml-cluster-select").value;
    loadMLClusterPlayers(state.mlPos, state.mlCluster);
  });

  $("#btn-ml-similar").addEventListener("click", loadMLSimilar);

  // Posición selector para similares
  $$("#ml-sim-pos-group .pos-radio").forEach(btn => {
    btn.addEventListener("click", () => {
      $$("#ml-sim-pos-group .pos-radio").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      loadMLRefPlayers(btn.dataset.pos);
    });
  });

  await loadMLClusters("Portero");
  await loadMLRefPlayers("Portero");
}

async function loadMLClusters(pos) {
  const data = await api(`/api/ml/clusters/${encodeURIComponent(pos)}`);
  if (!data.clusters?.length) {
    $("#ml-axis-label").textContent = "Sin datos de clustering";
    return;
  }

  $("#ml-axis-label").textContent = `${data.x_label} vs ${data.y_label}` +
    (data.var_explained ? ` (${data.var_explained}% varianza explicada)` : "");

  // Poblar selector de clusters
  const sel = $("#ml-cluster-select");
  sel.innerHTML = data.labels.map(l => `<option value="${l}">${l}</option>`).join("");
  state.mlCluster = data.labels[0];

  // Scatter
  renderMLScatter(data);

  // Cargar jugadores del primer cluster
  await loadMLClusterPlayers(pos, state.mlCluster);
}

function renderMLScatter(data) {
  destroyChart("mlScatter");
  if (typeof Chart === "undefined") return;
  const canvas = $("#ml-scatter");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const datasets = data.clusters.map((cl, i) => ({
    label: cl.label,
    data:  cl.players.map(p => ({ x: p.x_val, y: p.y_val, name: p.player, team: p.team })),
    backgroundColor: getClusterColors()[i % getClusterColors().length] + "BB",
    pointRadius: 6, pointHoverRadius: 9,
    borderWidth: 0,
  }));

  state.charts.mlScatter = new Chart(ctx, {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: C.text, boxWidth: 10, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const p = ctx.raw;
              return [`${p.name}`, `${p.team}`, `(${p.x?.toFixed(1)}, ${p.y?.toFixed(1)})`];
            }
          }
        }
      },
      scales: {
        x: { grid:{ color:C.border }, ticks:{ color:C.text2 },
             title:{ display:true, text: data.x_label, color:C.text3 } },
        y: { grid:{ color:C.border }, ticks:{ color:C.text2 },
             title:{ display:true, text: data.y_label, color:C.text3 } },
      }
    }
  });
}

async function loadMLClusterPlayers(pos, clusterLabel) {
  const players = await api(`/api/ml/cluster-players/${encodeURIComponent(pos)}/${encodeURIComponent(clusterLabel)}`);
  $("#ml-cluster-title").textContent = `Jugadores del perfil: ${clusterLabel}`;

  // Info del cluster
  const n = players.length;
  const avgGoals = (players.reduce((a,p) => a + (+p.goals||0), 0) / (n||1)).toFixed(1);
  const avgMin   = (players.reduce((a,p) => a + (+p.minutes||0), 0) / (n||1)).toFixed(0);
  $("#ml-cluster-info").innerHTML = `
    <div class="cluster-info-item">
      <span class="cluster-info-label">Jugadores</span>
      <span class="cluster-info-val">${n}</span>
    </div>
    <div class="cluster-info-item">
      <span class="cluster-info-label">Goles promedio</span>
      <span class="cluster-info-val">${avgGoals}</span>
    </div>
    <div class="cluster-info-item">
      <span class="cluster-info-label">Min. promedio</span>
      <span class="cluster-info-val">${avgMin}'</span>
    </div>`;

  // Tabla
  $("#ml-players-body").innerHTML = players.map(p => `
    <tr>
      <td class="player-name">${p.player}</td>
      <td>${p.team}</td>
      <td class="mono">${p.age_num > 0 ? Math.round(p.age_num) : "—"}</td>
      <td class="mono">${p.goals}</td>
      <td class="mono">${p.assists}</td>
      <td class="mono">${(+p.minutes||0).toLocaleString()}'</td>
      <td class="mono">${p.market_value_eur_fmt}</td>
    </tr>`).join("");
}

async function loadMLRefPlayers(pos) {
  const players = await api(`/api/ml/players/${encodeURIComponent(pos)}`);
  const refSel = $("#ml-ref-player");
  refSel.innerHTML = players.map(p => `<option value="${p}">${p}</option>`).join("");
}

async function loadMLSimilar() {
  const player   = $("#ml-ref-player").value;
  const n        = $("#ml-n").value;
  const exclTeam = $("#ml-excl").checked;
  if (!player) return;

  const info = await api(`/api/ml/player-info?player=${encodeURIComponent(player)}`);

  const refCard = $("#ml-ref-card");
  if (info.player) {
    const allData = await api(`/api/scout/${encodeURIComponent(info.team)}`);
    const pData = allData.plantel?.find(p => p.player === info.player) || {};
    pData.pos_group = info.pos_group;
    pData.market_value_eur_fmt = pData.market_value_eur || "N/D";
    refCard.innerHTML = `
      <div style="margin-bottom:8px;font-size:11px;color:${C.blue};font-family:'DM Mono';letter-spacing:1px">
        CLUSTER: ${info.cluster_label}
      </div>
      ${playerCard({...pData, player: info.player, team: info.team }, { gold: true })}`;
  }

  const url = `/api/ml/similar?player=${encodeURIComponent(player)}&n=${n}&excl_team=${exclTeam ? info.team : ""}`;
  const similar = await api(url);

  const grid = $("#ml-similar-grid");
  if (!similar.length) {
    grid.innerHTML = `<p style="color:${C.text3};grid-column:1/-1">No se encontraron jugadores similares.</p>`;
    return;
  }

  grid.innerHTML = similar.map(p => {
    const same = p.mismo_cluster;
    const clusterTag = { same, label: same ? "[MISMO PERFIL]" : "[PERFIL CERCANO]" };
    return playerCard(p, { showSim: true, clusterTag });
  }).join("");
}

// ══════════════════════════════════════════════
// VAEP
// ══════════════════════════════════════════════
async function loadVaep() {
  state.vaepLoaded = true;

  // Poblar equipos
  const teams = await api("/api/teams");
  const sel = $("#vaep-team");
  sel.innerHTML = `<option value="">Todos</option>` +
    teams.map(t => `<option value="${t}">${t}</option>`).join("");

  setupRangeDisplay("vaep-min", "vaep-min-val", v => `${v}'`);
  $("#btn-vaep").addEventListener("click", fetchAndRenderVaep);
  await fetchAndRenderVaep();
}

async function fetchAndRenderVaep() {
  const team   = $("#vaep-team").value;
  const pos    = $("#vaep-pos").value;
  const minMin = $("#vaep-min").value;
  const data   = await api(`/api/vaep?team=${encodeURIComponent(team)}&pos=${encodeURIComponent(pos)}&min_min=${minMin}`);

  if (!data.length) {
    $("#vaep-body").innerHTML = `<tr><td colspan="9" style="color:${C.text3};text-align:center">Sin datos de VAEP disponibles</td></tr>`;
    return;
  }

  state.vaepData = data;

  // Tabla
  $("#vaep-body").innerHTML = data.slice(0,50).map((p,i) => {
    const v = p.vaep_per90;
    const color = v > 0 ? C.green : C.red;
    return `<tr>
      <td class="mono" style="color:${C.text3}">${i+1}</td>
      <td class="player-name">${p.player}</td>
      <td>${p.team}</td>
      <td>${posBadge(p.pos_group)}</td>
      <td class="mono">${p.goals}</td>
      <td class="mono">${p.assists}</td>
      <td class="mono" style="color:${color}">${v?.toFixed(4)}</td>
      <td class="mono">${p.offensive_per90?.toFixed(4)}</td>
      <td class="mono">${p.defensive_per90?.toFixed(4)}</td>
    </tr>`;
  }).join("");

  // Chart top 10 total
  const top10 = data.slice(0,10);
  destroyChart("vaepTotal");
  state.charts.vaepTotal = new Chart($("#vaep-chart-total").getContext("2d"), {
    type: "bar",
    data: {
      labels: top10.map(p => p.player.split(" ").pop()),
      datasets: [{ data: top10.map(p => p.vaep_per90),
        backgroundColor: C.blue+"BB", borderRadius: 4, borderSkipped: false }]
    },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { legend:{ display:false } },
      scales: {
        x:{ grid:{ color:C.border }, ticks:{ color:C.text2 } },
        y:{ grid:{ color:C.border }, ticks:{ color:C.text } },
      }
    }
  });

  // Chart top 10 ofensivo
  const top10o = [...data].sort((a,b) => b.offensive_per90 - a.offensive_per90).slice(0,10);
  destroyChart("vaepOff");
  state.charts.vaepOff = new Chart($("#vaep-chart-off").getContext("2d"), {
    type: "bar",
    data: {
      labels: top10o.map(p => p.player.split(" ").pop()),
      datasets: [{ data: top10o.map(p => p.offensive_per90),
        backgroundColor: C.red+"BB", borderRadius: 4, borderSkipped: false }]
    },
    options: {
      indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { legend:{ display:false } },
      scales: {
        x:{ grid:{ color:C.border }, ticks:{ color:C.text2 } },
        y:{ grid:{ color:C.border }, ticks:{ color:C.text } },
      }
    }
  });

  // Scatter off vs def
  const byPos = {};
  data.forEach(p => {
    if (!byPos[p.pos_group]) byPos[p.pos_group] = [];
    byPos[p.pos_group].push(p);
  });
  destroyChart("vaepScatter");
  state.charts.vaepScatter = new Chart($("#vaep-scatter").getContext("2d"), {
    type: "scatter",
    data: {
      datasets: Object.entries(byPos).map(([pos, players]) => ({
        label: pos,
        data: players.map(p => ({ x: p.offensive_per90, y: p.defensive_per90, name: p.player, team: p.team })),
        backgroundColor: posColor(pos)+"99",
        pointRadius: 5, pointHoverRadius: 8,
      }))
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: C.text } },
        tooltip: {
          callbacks: { label: ctx => [`${ctx.raw.name}`, `${ctx.raw.team}`] }
        }
      },
      scales: {
        x: { grid:{ color:C.border }, ticks:{ color:C.text2 },
             title:{ display:true, text:"VAEP Ofensivo/90", color:C.text3 } },
        y: { grid:{ color:C.border }, ticks:{ color:C.text2 },
             title:{ display:true, text:"VAEP Defensivo/90", color:C.text3 } },
      }
    }
  });
}

// ══════════════════════════════════════════════
// LIGA
// ══════════════════════════════════════════════
async function loadLiga() {
  state.ligaLoaded = true;

  // Stats generales
  const stats = await api("/api/liga/stats");
  $("#liga-metrics").innerHTML = Object.entries({
    "Jugadores":     [stats.jugadores,   null],
    "Equipos":       [stats.equipos,     C.blue],
    "Goles totales": [stats.goles,       C.green],
    "Valor promedio":[stats.valor_prom,  C.purple],
  }).map(([lbl,[val,color]]) => `
    <div class="metric-card">
      <span class="metric-val" style="color:${color||C.blue}">${val}</span>
      <span class="metric-lbl">${lbl}</span>
    </div>`).join("");

  // Tabla de posiciones
  const standings = await api("/api/liga/standings");
  renderStandings("body-zona-a", standings.zona_a);
  renderStandings("body-zona-b", standings.zona_b);

  // Top goleadores
  const scorers = await api("/api/liga/top-scorers");
  destroyChart("ligaScorers");
  state.charts.ligaScorers = new Chart($("#liga-scorers").getContext("2d"), {
    type: "bar",
    data: {
      labels: scorers.map(p => p.player.split(" ").pop()),
      datasets: [
        { label:"Goles", data:scorers.map(p=>p.goals),
          backgroundColor:C.blue+"BB", borderRadius:4, borderSkipped:false },
        { label:"Asistencias", data:scorers.map(p=>p.assists),
          backgroundColor:C.gold+"BB", borderRadius:4, borderSkipped:false },
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{ labels:{ color:C.text } } },
      scales:{
        x:{ grid:{ color:C.border }, ticks:{ color:C.text } },
        y:{ grid:{ color:C.border }, ticks:{ color:C.text2 } },
      }
    }
  });

  // Valores por equipo
  const values = await api("/api/liga/team-values");
  destroyChart("ligaValues");
  state.charts.ligaValues = new Chart($("#liga-values").getContext("2d"), {
    type:"bar",
    data:{
      labels: values.map(t=>t.team),
      datasets:[{ label:"Valor (€)", data:values.map(t=>t.value),
        backgroundColor:C.purple+"BB", borderRadius:4, borderSkipped:false }]
    },
    options:{
      indexAxis:"y", responsive:true, maintainAspectRatio:false,
      plugins:{
        legend:{ display:false },
        tooltip:{ callbacks:{ label: ctx => values[ctx.dataIndex].value_fmt } }
      },
      scales:{
        x:{ grid:{ color:C.border }, ticks:{ color:C.text2, callback: v => v>=1e6?`€${(v/1e6).toFixed(0)}M`:v } },
        y:{ grid:{ color:C.border }, ticks:{ color:C.text } },
      }
    }
  });
}

function renderStandings(bodyId, data) {
  const n = data.length;
  const body = $(`#${bodyId}`);
  body.innerHTML = data.map((row, i) => {
    const pos = i + 1;
    let cls = "";
    if (pos <= 8)  cls = "clasif";
    if (pos >= n-1) cls = "descenso";
    const dg = row.DG > 0 ? `+${row.DG}` : row.DG;
    const shieldUrl = getTeamShield(row.team);
    return `<tr class="${cls}">
      <td class="mono" style="color:${C.text3}">${pos}</td>
      <td class="player-name"><img src="${shieldUrl}" alt="" style="width:18px;height:18px;vertical-align:middle;margin-right:6px;object-fit:contain" onerror="this.style.display='none'">${row.team}</td>
      <td class="mono">${row.PJ}</td>
      <td class="mono text-green">${row.G}</td>
      <td class="mono">${row.E}</td>
      <td class="mono text-red">${row.P}</td>
      <td class="mono">${row.GF}</td>
      <td class="mono">${row.GC}</td>
      <td class="mono" style="color:${row.DG>=0?C.green:C.red}">${dg}</td>
      <td class="mono">${row.VI}</td>
      <td class="mono pts" style="color:${C.blue};font-weight:600">${row.Pts}</td>
    </tr>`;
  }).join("");
}

// ══════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  const stored = localStorage.getItem("theme");
  function applyTheme(isDark) {
    if (isDark) {
      document.documentElement.classList.add('dark-theme');
    } else {
      document.documentElement.classList.remove('dark-theme');
    }
    updateJSColors(isDark);
    const sun = document.querySelector('.theme-icon-sun');
    const moon = document.querySelector('.theme-icon-moon');
    if (sun && moon) {
      sun.style.display = isDark ? 'none' : '';
      moon.style.display = isDark ? '' : 'none';
    }
  }

  const darkModeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
  if (stored !== null) {
    applyTheme(stored === "dark");
  } else {
    applyTheme(darkModeMediaQuery.matches);
  }
  darkModeMediaQuery.addEventListener('change', e => {
    if (localStorage.getItem("theme") !== null) return;
    applyTheme(e.matches);
  });

  const toggleBtn = document.getElementById('theme-toggle');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const isDark = !document.documentElement.classList.contains('dark-theme');
      localStorage.setItem("theme", isDark ? "dark" : "light");
      applyTheme(isDark);
    });
  }

  const page = CURRENT_PAGE;
  if (page === "/scout" || page === "") {
    initScout();
  } else if (page === "/ml") {
    initML();
  } else if (page === "/vaep") {
    loadVaep();
  } else if (page === "/liga") {
    loadLiga();
  }
});
