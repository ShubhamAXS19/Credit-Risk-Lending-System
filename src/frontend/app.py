"""
Credit Risk & Lending System — Unified Frontend
================================================
Merges:
  - Multi-project sidebar navigation
  - Credit Risk Scorecard Dashboard (dark theme)
  - Live borrower scoring form with API integration

Install:
    pip install streamlit plotly pandas numpy requests

Run:
    streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# =============================================================================
# PAGE CONFIG  (must be first Streamlit call)
# =============================================================================
st.set_page_config(
    page_title="Credit Risk & Lending System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# GLOBAL CSS — dark theme, IBM Plex fonts, custom components
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
  font-family: 'IBM Plex Sans', sans-serif !important;
  background-color: #1a1c18 !important;
  color: #d4d0c4 !important;
}
.stApp { background-color: #1a1c18; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background-color: #141612 !important;
  border-right: 0.5px solid #2e2e28;
}
[data-testid="stSidebar"] * { color: #d4d0c4 !important; }
[data-testid="stSidebarNav"] { display: none; }
.sidebar-logo {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 13px;
  font-weight: 500;
  color: #4caf78 !important;
  letter-spacing: 0.08em;
  padding: 0 0 20px 0;
  border-bottom: 0.5px solid #2e2e28;
  margin-bottom: 20px;
}
.nav-section {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #5a5a50 !important;
  margin: 20px 0 8px 0;
}

/* ── Streamlit radio overrides for nav look ── */
[data-testid="stSidebar"] .stRadio label {
  font-size: 13px !important;
  color: #8a8878 !important;
  padding: 5px 8px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: #222420;
  color: #d4d0c4 !important;
}
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + div label,
[data-testid="stSidebar"] div[data-baseweb="radio"] input:checked ~ label {
  color: #d4d0c4 !important;
}

/* ── Page title ── */
.page-title {
  font-family: 'IBM Plex Sans', sans-serif;
  font-size: 22px;
  font-weight: 500;
  color: #d4d0c4;
  margin: 0 0 4px 0;
  line-height: 1.2;
}
.page-subtitle {
  font-size: 13px;
  color: #5a5a50;
  margin: 0 0 28px 0;
}

/* ── Section titles ── */
.section-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #5a5a50;
  margin: 0 0 14px 0;
  padding: 0;
}

/* ── Metric card ── */
.metric-card {
  background: #222420;
  border: 0.5px solid #2e2e28;
  border-radius: 10px;
  padding: 16px 18px;
  height: 100%;
}
.metric-label { font-size: 12px; color: #8a8878; margin: 0 0 4px 0; }
.metric-value {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 26px;
  font-weight: 500;
  margin: 0 0 3px 0;
  line-height: 1.1;
}
.metric-value.good { color: #4caf78; }
.metric-value.warn { color: #d4a843; }
.metric-value.bad  { color: #e05a4a; }
.metric-sub { font-size: 11px; color: #5a5a50; margin: 0; }

/* ── Chart card wrapper ── */
.chart-card {
  background: #1e2019;
  border: 0.5px solid #2e2e28;
  border-radius: 12px;
  padding: 18px 20px;
}

/* ── Decision cards ── */
.decision-card { border-radius: 10px; padding: 16px 18px; border: 0.5px solid; }
.decision-card.approve { background: #0f2418; border-color: #2a6642; }
.decision-card.review  { background: #221c0c; border-color: #5a4010; }
.decision-card.reject  { background: #1f0e0e; border-color: #6a2020; }
.decision-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px; font-weight: 500;
  letter-spacing: 0.12em; text-transform: uppercase;
  margin: 0 0 6px 0;
}
.decision-label.approve { color: #4caf78; }
.decision-label.review  { color: #d4a843; }
.decision-label.reject  { color: #e05a4a; }
.decision-n {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 20px; font-weight: 500; color: #d4d0c4; margin: 0 0 4px 0;
}
.decision-pct { font-size: 13px; color: #8a8878; font-weight: 400; }
.decision-meta { font-size: 12px; color: #5a5a50; margin: 3px 0 0 0; }

/* ── Scorecard table ── */
.sc-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.sc-table th {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px; font-weight: 500; color: #8a8878;
  text-align: left; padding: 8px 10px;
  border-bottom: 0.5px solid #2e2e28; letter-spacing: 0.04em;
}
.sc-table td { padding: 6px 10px; border-bottom: 0.5px solid #252520; color: #c8c4b4; }
.sc-table td.pts { font-family: 'IBM Plex Mono', monospace; font-weight: 500; text-align: right; }
.pts-pos { color: #4caf78; }
.pts-neg { color: #e05a4a; }
.feat-header { background: #252520; }
.feat-header td {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px; font-weight: 500; color: #6a6a60;
  letter-spacing: 0.08em; text-transform: uppercase; padding: 5px 10px;
}
.sc-table tr:last-child td { border-bottom: none; }

/* ── Borrower cards ── */
.borrower-card {
  background: #1e2019;
  border: 0.5px solid #2e2e28;
  border-radius: 12px;
  padding: 16px 18px;
  border-left-width: 3px;
}
.borrower-card.approve { border-left-color: #4caf78; }
.borrower-card.reject  { border-left-color: #e05a4a; }
.borrower-title { font-size: 14px; font-weight: 500; color: #d4d0c4; margin: 0 0 12px 0; }
.badge {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px; font-weight: 500;
  padding: 2px 8px; border-radius: 10px;
  margin-left: 8px; vertical-align: middle; letter-spacing: 0.06em;
}
.badge.approve { background: #0f2418; color: #4caf78; border: 0.5px solid #2a6642; }
.badge.reject  { background: #1f0e0e; color: #e05a4a; border: 0.5px solid #6a2020; }
.bstat {
  display: flex; justify-content: space-between;
  font-size: 12px; padding: 5px 0;
  border-bottom: 0.5px solid #252520;
}
.bstat-label { color: #8a8878; }
.bstat-val { font-weight: 500; color: #d4d0c4; font-family: 'IBM Plex Mono', monospace; }
.woe-section-title { font-size: 12px; color: #5a5a50; margin: 14px 0 8px 0; }
.woe-row { display: flex; align-items: center; gap: 8px; margin: 5px 0; font-size: 11px; }
.woe-feat { width: 155px; color: #8a8878; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.woe-bar-wrap { flex: 1; height: 6px; background: #2a2a24; border-radius: 3px; position: relative; overflow: hidden; }
.woe-bar { height: 100%; border-radius: 3px; position: absolute; }
.woe-bar.pos { background: #4caf78; }
.woe-bar.neg { background: #e05a4a; right: 0; }
.woe-val { width: 50px; text-align: right; font-family: 'IBM Plex Mono', monospace; color: #8a8878; }

/* ── Live scorer result cards ── */
.result-card {
  border-radius: 10px; padding: 20px 22px;
  border: 0.5px solid; margin-top: 4px;
}
.result-card.approve { background: #0f2418; border-color: #2a6642; }
.result-card.review  { background: #221c0c; border-color: #5a4010; }
.result-card.reject  { background: #1f0e0e; border-color: #6a2020; }
.result-decision {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 18px; font-weight: 500; letter-spacing: 0.08em;
  text-transform: uppercase; margin: 0 0 6px 0;
}
.result-decision.approve { color: #4caf78; }
.result-decision.review  { color: #d4a843; }
.result-decision.reject  { color: #e05a4a; }
.result-score {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 36px; font-weight: 500; color: #d4d0c4;
  margin: 8px 0 4px 0; line-height: 1;
}
.result-pd { font-size: 13px; color: #8a8878; margin: 0; }

/* ── Info box ── */
.info-box {
  background: #1a2235;
  border: 0.5px solid #1e3a5f;
  border-radius: 8px;
  padding: 12px 16px;
  font-size: 12px;
  color: #6a9fd4;
  margin-bottom: 20px;
}

/* ── Under construction ── */
.wip-card {
  background: #1e2019;
  border: 0.5px solid #2e2e28;
  border-radius: 12px;
  padding: 48px 32px;
  text-align: center;
  margin-top: 20px;
}
.wip-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px; letter-spacing: 0.12em;
  text-transform: uppercase; color: #5a5a50;
  margin: 0 0 12px 0;
}
.wip-title { font-size: 18px; font-weight: 500; color: #d4d0c4; margin: 0 0 8px 0; }
.wip-sub { font-size: 13px; color: #5a5a50; margin: 0; }

/* ── Streamlit form overrides ── */
.stTextInput input, .stNumberInput input, .stSelectbox select {
  background: #222420 !important;
  border: 0.5px solid #3e3e38 !important;
  color: #d4d0c4 !important;
  border-radius: 6px !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
}
.stSlider [data-baseweb="slider"] { background: transparent !important; }
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { color: #5a5a50 !important; }
.stFormSubmitButton button {
  background: #1a2e1e !important;
  border: 0.5px solid #2a6642 !important;
  color: #4caf78 !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 12px !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  border-radius: 6px !important;
  padding: 8px 24px !important;
  transition: background 0.15s !important;
}
.stFormSubmitButton button:hover {
  background: #1f3824 !important;
}
label[data-testid="stWidgetLabel"] {
  font-size: 12px !important;
  color: #8a8878 !important;
}

/* ── Divider ── */
.divider { border: none; border-top: 0.5px solid #2e2e28; margin: 28px 0; }

/* ── Home page project grid ── */
.project-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 8px; }
.project-tile {
  background: #1e2019;
  border: 0.5px solid #2e2e28;
  border-radius: 12px;
  padding: 18px 20px;
  transition: border-color 0.2s;
}
.project-tile:hover { border-color: #4a4a3e; }
.project-tile-num {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px; font-weight: 500; color: #4caf78;
  letter-spacing: 0.1em; margin: 0 0 8px 0;
}
.project-tile-title { font-size: 14px; font-weight: 500; color: #d4d0c4; margin: 0 0 6px 0; }
.project-tile-desc { font-size: 12px; color: #5a5a50; margin: 0; line-height: 1.5; }
.project-tile-status {
  display: inline-block;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px; font-weight: 500; letter-spacing: 0.08em;
  text-transform: uppercase; padding: 2px 7px;
  border-radius: 8px; margin-top: 12px;
}
.status-live { background: #0f2418; color: #4caf78; border: 0.5px solid #2a6642; }
.status-wip  { background: #252520; color: #5a5a50; border: 0.5px solid #3e3e38; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# STATIC DATA  (replace with live pipeline outputs as needed)
# =============================================================================

LIFT_DATA = {
    'decile':   ['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10'],
    'lift':     [5.40, 3.59, 2.73, 2.20, 1.85, 1.59, 1.39, 1.23, 1.11, 1.00],
    'bad_rate': [36.1, 11.9, 6.7, 4.0, 3.0, 2.1, 1.1, 0.8, 0.7, 0.4],
}

SCORE_BAND_DATA = {
    'band':    ['366–397','397–428','428–459','459–489','489–520','520–551'],
    'default': [60.3, 42.5, 24.1, 10.3, 4.2, 1.0],
    'pop_pct': [1.6, 3.2, 5.4, 14.0, 25.0, 50.7],
}

SCORECARD = [
    ("TOTAL DELINQUENCY",       None, None, None, None),
    (None, "0 incidents",       "2.7%",  "+0.933",  23.5),
    (None, "1 incident",        "12.2%", "-0.659",  -9.1),
    (None, "2 incidents",       "23.9%", "-1.480", -26.0),
    (None, "3–5 incidents",     "40.1%", "-2.233", -41.4),
    (None, ">5 incidents",      "59.9%", "-3.000", -57.2),
    ("REVOLVING UTILIZATION",   None, None, None, None),
    (None, "< 10%",             "1.8%",  "+1.341",  30.1),
    (None, "10–30%",            "3.1%",  "+0.804",  19.8),
    (None, "30–50%",            "5.7%",  "+0.162",   7.5),
    (None, "50–70%",            "9.6%",  "-0.396",  -3.2),
    (None, "70–90%",            "15.2%", "-0.915", -13.1),
    (None, "> 90%",             "22.3%", "-1.389", -22.2),
    ("90-DAY DELINQUENCIES",    None, None, None, None),
    (None, "None",              "4.6%",  "+0.393",   7.2),
    (None, "1 time",            "33.8%", "-1.965",  -9.6),
    (None, "2 times",           "49.5%", "-2.617", -14.3),
    (None, ">2 times",          "60.8%", "-3.000", -17.0),
    ("AGE",                     None, None, None, None),
    (None, "< 25",              "11.0%", "-0.545",  -2.9),
    (None, "25–35",             "10.9%", "-0.532",  -2.8),
    (None, "35–45",             "8.8%",  "-0.300",   0.4),
    (None, "45–55",             "7.6%",  "-0.140",   2.5),
    (None, "55–65",             "4.7%",  "+0.379",   9.5),
    (None, "> 65",              "2.5%",  "+1.042",  18.4),
    ("INCOME PER PERSON",       None, None, None, None),
    (None, "< $1,827",          "10.2%", "-0.459",   1.0),
    (None, "$1,827–$3,076",     "7.3%",  "-0.093",   3.7),
    (None, "$3,076–$5,125",     "6.0%",  "+0.120",   5.3),
    (None, "$5,125–$5,650",     "5.4%",  "+0.223",   6.0),
    (None, "> $5,650",          "4.6%",  "+0.406",   7.4),
]

MAX_WOE = 3.0

EXAMPLE_BORROWERS = {
    "A": dict(
        title="Borrower A — Low Risk", decision="approve",
        score=533.0, pd="16.94%", age=52, util="5%", lates=0, income="$8,000",
        woe=[("total_delinquency", 0.9328), ("revolving_util", 1.3406),
             ("90days_late", 0.3929), ("age", -0.1402), ("income_per_person", 0.1197)],
    ),
    "B": dict(
        title="Borrower B — High Risk", decision="reject",
        score=368.8, pd="98.37%", age=24, util="92%", lates=2, income="$2,200",
        woe=[("total_delinquency", -3.0000), ("revolving_util", -1.3889),
             ("90days_late", -2.6171), ("age", -0.5448), ("income_per_person", -0.4592)],
    ),
}


# =============================================================================
# PLOTLY CHART HELPERS
# =============================================================================

_BG   = "#1e2019"
_GRID = "#2e2e28"
_FONT = "#8a8878"
_TICK = "#6a6a60"

def _base_layout(title="", height=240):
    return dict(
        title=dict(text=title, font=dict(size=13, color="#d4d0c4", family="IBM Plex Sans"), x=0, y=1, pad=dict(b=14)),
        plot_bgcolor=_BG, paper_bgcolor=_BG,
        font=dict(family="IBM Plex Sans", color=_FONT, size=11),
        height=height,
        margin=dict(l=8, r=8, t=40, b=8),
        showlegend=False,
        xaxis=dict(gridcolor=_GRID, gridwidth=0.5, tickfont=dict(size=11, color=_TICK),
                   linecolor=_GRID, tickcolor=_GRID),
        yaxis=dict(gridcolor=_GRID, gridwidth=0.5, tickfont=dict(size=11, color=_TICK),
                   linecolor=_GRID, tickcolor=_GRID),
        hoverlabel=dict(bgcolor="#2a2a24", bordercolor="#3e3e38",
                        font=dict(family="IBM Plex Mono", size=12, color="#d4d0c4")),
    )

def lift_chart():
    df = pd.DataFrame(LIFT_DATA)
    colors = ["#e05a4a" if v >= 3 else "#d4a843" if v >= 2 else "#4a4a3e" for v in df["lift"]]
    fig = go.Figure(go.Bar(
        x=df["decile"], y=df["lift"],
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Lift: %{y:.2f}×<extra></extra>",
        width=0.65,
    ))
    lo = _base_layout("Lift by decile")
    lo["yaxis"]["ticksuffix"] = "×"
    lo["yaxis"]["range"] = [0, 6.2]
    fig.update_layout(**lo)
    fig.update_traces(marker_cornerradius=3)
    return fig

def default_rate_chart():
    df = pd.DataFrame(SCORE_BAND_DATA)
    colors = ["#e05a4a" if v >= 20 else "#d4a843" if v >= 10 else "#4caf78" for v in df["default"]]
    fig = go.Figure(go.Bar(
        x=df["band"], y=df["default"],
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>Default rate: %{y:.1f}%<extra></extra>",
        width=0.65,
    ))
    lo = _base_layout("Default rate by score band")
    lo["yaxis"]["ticksuffix"] = "%"
    lo["yaxis"]["range"] = [0, 72]
    lo["xaxis"]["tickangle"] = -20
    lo["xaxis"]["tickfont"]["size"] = 10
    fig.update_layout(**lo)
    fig.update_traces(marker_cornerradius=3)
    return fig

def score_gauge(score, score_min=366, score_max=551):
    """Gauge chart for a single borrower's credit score."""
    norm = (score - score_min) / (score_max - score_min)
    color = "#4caf78" if norm >= 0.7 else "#d4a843" if norm >= 0.4 else "#e05a4a"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(family="IBM Plex Mono", size=32, color="#d4d0c4")),
        gauge=dict(
            axis=dict(range=[score_min, score_max],
                      tickfont=dict(size=10, color=_TICK),
                      tickcolor=_GRID),
            bar=dict(color=color, thickness=0.25),
            bgcolor="#252520",
            borderwidth=0,
            steps=[
                dict(range=[score_min, score_min + (score_max-score_min)*0.4], color="#2a1a1a"),
                dict(range=[score_min + (score_max-score_min)*0.4,
                             score_min + (score_max-score_min)*0.7], color="#2a2210"),
                dict(range=[score_min + (score_max-score_min)*0.7, score_max], color="#0f2418"),
            ],
        ),
    ))
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        height=180,
        margin=dict(l=20, r=20, t=10, b=0),
        font=dict(family="IBM Plex Sans", color=_FONT),
    )
    return fig


# =============================================================================
# HTML COMPONENT BUILDERS
# =============================================================================

def metric_card(label, value, cls, sub):
    return f"""<div class="metric-card">
  <p class="metric-label">{label}</p>
  <p class="metric-value {cls}">{value}</p>
  <p class="metric-sub">{sub}</p>
</div>"""

def decision_card(cls, label, n, pct, dr, score_range):
    return f"""<div class="decision-card {cls}">
  <p class="decision-label {cls}">{label}</p>
  <p class="decision-n">{n} <span class="decision-pct">({pct})</span></p>
  <p class="decision-meta">Default rate: {dr}</p>
  <p class="decision-meta">Score: {score_range}</p>
</div>"""

def scorecard_table_html():
    rows = ""
    for (feat, bin_label, bad_rate, woe, pts) in SCORECARD:
        if feat is not None:
            rows += f'<tr class="feat-header"><td colspan="4">{feat}</td></tr>\n'
        else:
            sign  = "+" if pts > 0 else ""
            color = "pts-pos" if pts >= 0 else "pts-neg"
            rows += f"""<tr>
  <td style="padding-left:20px">{bin_label}</td>
  <td>{bad_rate}</td>
  <td style="font-family:'IBM Plex Mono',monospace">{woe}</td>
  <td class="pts"><span class="{color}">{sign}{pts}</span></td>
</tr>\n"""
    return f"""<div class="chart-card">
  <table class="sc-table">
    <thead><tr>
      <th>Feature / Bin</th><th>Bad rate</th>
      <th>WOE</th><th style="text-align:right">Points</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""

def _woe_bar(feat, val):
    pct  = abs(val) / MAX_WOE * 100
    side = "neg" if val < 0 else "pos"
    bar  = f'<div class="woe-bar {side}" style="width:{pct:.0f}%"></div>'
    sign = "" if val < 0 else "+"
    return f"""<div class="woe-row">
  <span class="woe-feat">{feat}</span>
  <div class="woe-bar-wrap">{bar}</div>
  <span class="woe-val">{sign}{val:.3f}</span>
</div>"""

def borrower_card_html(data):
    d        = data["decision"]
    woe_rows = "".join(_woe_bar(f, v) for f, v in data["woe"])
    return f"""<div class="borrower-card {d}">
  <p class="borrower-title">{data['title']}
    <span class="badge {d}">{d.upper()}</span>
  </p>
  <div class="bstat"><span class="bstat-label">Credit score</span>
    <span class="bstat-val">{data['score']}</span></div>
  <div class="bstat"><span class="bstat-label">PD probability</span>
    <span class="bstat-val">{data['pd']}</span></div>
  <div class="bstat"><span class="bstat-label">Age</span>
    <span class="bstat-val">{data['age']}</span></div>
  <div class="bstat"><span class="bstat-label">Utilization</span>
    <span class="bstat-val">{data['util']}</span></div>
  <div class="bstat"><span class="bstat-label">90-day lates</span>
    <span class="bstat-val">{data['lates']}</span></div>
  <div class="bstat" style="border-bottom:none">
    <span class="bstat-label">Monthly income</span>
    <span class="bstat-val">{data['income']}</span></div>
  <p class="woe-section-title">WOE contributions</p>
  {woe_rows}
</div>"""


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

with st.sidebar:
    st.markdown('<p class="sidebar-logo">🏦 CREDIT RISK SYSTEM</p>', unsafe_allow_html=True)

    st.markdown('<p class="nav-section">Projects</p>', unsafe_allow_html=True)
    page = st.radio(
        label="nav",
        options=[
            "🏠  Home",
            "1 · Credit Default Prediction",
            "2 · Loan Amount Prediction",
            "3 · Early Payment Prediction",
            "4 · Credit Score Prediction",
            "5 · Loan Recovery Prediction",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<p class="nav-section">Model Info</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:11px;color:#5a5a50;line-height:1.6">'
        'Dataset: Give Me Some Credit<br>'
        'Algorithm: Logistic Regression<br>'
        'Features: WOE-transformed<br>'
        'AUC: 0.860 &nbsp;|&nbsp; KS: 56.3'
        '</p>',
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE: HOME
# =============================================================================

if page == "🏠  Home":
    st.markdown('<p class="page-title">Credit Risk & Lending System</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">End-to-end credit risk pipeline — '
        'scorecard modelling, lending decisions, and portfolio monitoring.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-title">Available Modules</p>', unsafe_allow_html=True)
    st.markdown("""
<div class="project-grid">
  <div class="project-tile">
    <p class="project-tile-num">PROJECT 01</p>
    <p class="project-tile-title">Credit Default Prediction</p>
    <p class="project-tile-desc">WOE logistic regression scorecard. Predicts probability of default within 2 years.</p>
    <span class="project-tile-status status-live">Live</span>
  </div>
  <div class="project-tile">
    <p class="project-tile-num">PROJECT 02</p>
    <p class="project-tile-title">Loan Amount Prediction</p>
    <p class="project-tile-desc">Regression model estimating the appropriate loan amount for a borrower profile.</p>
    <span class="project-tile-status status-wip">Coming soon</span>
  </div>
  <div class="project-tile">
    <p class="project-tile-num">PROJECT 03</p>
    <p class="project-tile-title">Early Payment Prediction</p>
    <p class="project-tile-desc">Forecast prepayment risk and expected loan duration for portfolio planning.</p>
    <span class="project-tile-status status-wip">Coming soon</span>
  </div>
  <div class="project-tile">
    <p class="project-tile-num">PROJECT 04</p>
    <p class="project-tile-title">Credit Score Prediction</p>
    <p class="project-tile-desc">AI-driven credit scoring using alternative data sources beyond bureau scores.</p>
    <span class="project-tile-status status-wip">Coming soon</span>
  </div>
  <div class="project-tile">
    <p class="project-tile-num">PROJECT 05</p>
    <p class="project-tile-title">Loan Recovery Prediction</p>
    <p class="project-tile-desc">Estimate recovery amounts and timelines for defaulted loans in the portfolio.</p>
    <span class="project-tile-status status-wip">Coming soon</span>
  </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE: PROJECT 1 — CREDIT DEFAULT PREDICTION
# =============================================================================

elif page == "1 · Credit Default Prediction":
    st.markdown('<p class="page-title">Credit Default Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Give Me Some Credit dataset · '
        'WOE logistic regression scorecard · AUC 0.860 · KS 56.3</p>',
        unsafe_allow_html=True,
    )

    # ── Tabs: Dashboard | Live Scorer ────────────────────────────────────────
    tab_dash, tab_score = st.tabs(["📊  Model Dashboard", "🎯  Live Borrower Scorer"])

    # ── TAB 1: DASHBOARD ─────────────────────────────────────────────────────
    with tab_dash:

        st.markdown("<br>", unsafe_allow_html=True)

        # Section 1 — Performance metrics
        st.markdown('<p class="section-title">Model Performance — Test Set</p>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4, gap="small")
        with m1: st.markdown(metric_card("AUC", "0.860", "good", "Strong (>0.80 threshold)"), unsafe_allow_html=True)
        with m2: st.markdown(metric_card("KS statistic", "56.3", "good", "Strong (>40 threshold)"), unsafe_allow_html=True)
        with m3: st.markdown(metric_card("Decile 1 lift", "5.40×", "good", "36.1% bad rate at top decile"), unsafe_allow_html=True)
        with m4: st.markdown(metric_card("Top 3 decile capture", "81.8%", "good", "Bads caught in top 30%"), unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Section 2 — Charts
        st.markdown('<p class="section-title">Score Distribution & Default Rate by Band</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(lift_chart(), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(default_rate_chart(), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Section 3 — Decision strategy
        st.markdown('<p class="section-title">Lending Decision Strategy (p70 / p40 Thresholds)</p>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3, gap="medium")
        with d1: st.markdown(decision_card("approve", "APPROVE",       "9,212",  "30.7%", "0.7%",  "≥ 531.8"),         unsafe_allow_html=True)
        with d2: st.markdown(decision_card("review",  "MANUAL REVIEW", "8,803",  "29.3%", "2.0%",  "510.4 – 531.8"),   unsafe_allow_html=True)
        with d3: st.markdown(decision_card("reject",  "REJECT",        "11,985", "40.0%", "14.7%", "< 510.4"),         unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Section 4 — Scorecard table
        st.markdown('<p class="section-title">Credit Scorecard — Feature × Bin Points</p>', unsafe_allow_html=True)
        st.markdown(scorecard_table_html(), unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # Section 5 — Example borrowers
        st.markdown('<p class="section-title">Example Borrower Scoring</p>', unsafe_allow_html=True)
        b1, b2 = st.columns(2, gap="medium")
        with b1: st.markdown(borrower_card_html(EXAMPLE_BORROWERS["A"]), unsafe_allow_html=True)
        with b2: st.markdown(borrower_card_html(EXAMPLE_BORROWERS["B"]), unsafe_allow_html=True)

    # ── TAB 2: LIVE SCORER ───────────────────────────────────────────────────
    with tab_score:

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div class="info-box">Enter borrower details below. '
            'The model scores in real time using the trained WOE logistic regression pipeline. '
            'API endpoint: <code>POST /api/v1/predict</code></div>',
            unsafe_allow_html=True,
        )

        with st.form("live_scorer"):
            st.markdown('<p class="section-title">Borrower Profile</p>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3, gap="medium")

            with col1:
                st.markdown("**Financial**")
                revolving_util = st.slider(
                    "Revolving utilization", 0.0, 1.0, 0.15, step=0.01,
                    help="Fraction of total unsecured credit limit currently used"
                )
                debt_ratio = st.slider(
                    "Debt ratio", 0.0, 1.0, 0.25, step=0.01,
                    help="Monthly debt payments ÷ monthly income"
                )
                monthly_income = st.number_input(
                    "Monthly income ($)", value=6000, step=100, min_value=0,
                )

            with col2:
                st.markdown("**Delinquency history**")
                times_30_59 = st.number_input("30–59 days late (count)", value=0, step=1, min_value=0)
                times_60_89 = st.number_input("60–89 days late (count)", value=0, step=1, min_value=0)
                times_90    = st.number_input("90+ days late (count)",   value=0, step=1, min_value=0)

            with col3:
                st.markdown("**Demographics**")
                age             = st.number_input("Age (years)",         value=40, step=1, min_value=18, max_value=100)
                num_dependents  = st.number_input("Number of dependents", value=1, step=1, min_value=0)
                open_credit     = st.number_input("Open credit lines",    value=8, step=1, min_value=0)
                real_estate     = st.number_input("Real estate loans",    value=1, step=1, min_value=0)

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("→  Score this borrower")

        if submitted:
            # ── Feature engineering (mirrors the pipeline) ──────────────────
            total_delinquency = times_30_59 + times_60_89 + times_90
            income_per_person = monthly_income / (num_dependents + 1)
            credit_stress     = revolving_util * debt_ratio

            # ── WOE lookup tables (from trained model) ───────────────────────
            def lookup_woe(feature, value):
                tables = {
                    "total_delinquency": [
                        (0,  0,   0.9328),
                        (1,  1,  -0.6586),
                        (2,  2,  -1.4795),
                        (3,  5,  -2.2331),
                        (6, 999, -3.0000),
                    ],
                    "revolvingutilization": [
                        (0.000, 0.099, 1.3406),
                        (0.100, 0.299, 0.8038),
                        (0.300, 0.499, 0.1615),
                        (0.500, 0.699,-0.3963),
                        (0.700, 0.899,-0.9152),
                        (0.900, 9.999,-1.3889),
                    ],
                    "numberoftimes90dayslate": [
                        (0,  0,   0.3929),
                        (1,  1,  -1.9654),
                        (2,  2,  -2.6171),
                        (3, 999, -3.0000),
                    ],
                    "age": [
                        (0,  24, -0.5448),
                        (25, 34, -0.5320),
                        (35, 44, -0.2995),
                        (45, 54, -0.1402),
                        (55, 64,  0.3792),
                        (65, 999, 1.0422),
                    ],
                    "income_per_person": [
                        (0,       1827.27, -0.4592),
                        (1827.27, 3076.0,  -0.0927),
                        (3076.0,  5125.0,   0.1197),
                        (5125.0,  5650.0,   0.2234),
                        (5650.0,  999999,   0.4057),
                    ],
                }
                for lo, hi, woe in tables[feature]:
                    if lo <= value <= hi:
                        return woe
                # fallback: median
                woes = [w for _, _, w in tables[feature]]
                return sorted(woes)[len(woes)//2]

            woe_td   = lookup_woe("total_delinquency",       total_delinquency)
            woe_util = lookup_woe("revolvingutilization",     revolving_util)
            woe_90   = lookup_woe("numberoftimes90dayslate",  times_90)
            woe_age  = lookup_woe("age",                      age)
            woe_inc  = lookup_woe("income_per_person",        income_per_person)

            # ── Logistic regression score (coefficients from trained model) ──
            # logit(PD) = intercept + sum(coef_i * woe_i)
            # Coefficients are scaled back from StandardScaler
            INTERCEPT = -0.7602
            # original (unscaled) coefficients:
            coefs = {
                "td":   -0.664456,
                "util": -0.663910,
                "90":   -0.153571,
                "age":  -0.250578,
                "inc":  -0.076264,
            }
            # StandardScaler std values from training
            stds = {"td": 0.9342, "util": 1.0003, "90": 0.6210, "age": 0.5382, "inc": 0.2968}

            logit = INTERCEPT
            logit += (coefs["td"]   / stds["td"])   * woe_td
            logit += (coefs["util"] / stds["util"])  * woe_util
            logit += (coefs["90"]   / stds["90"])    * woe_90
            logit += (coefs["age"]  / stds["age"])   * woe_age
            logit += (coefs["inc"]  / stds["inc"])   * woe_inc

            pd_prob = 1 / (1 + np.exp(-logit))

            # ── Scorecard scaling ─────────────────────────────────────────────
            FACTOR = 20 / np.log(2)
            OFFSET = 600 - FACTOR * np.log(50)
            odds   = max((1 - pd_prob) / pd_prob, 1e-6)
            score  = OFFSET + FACTOR * np.log(odds)

            # ── Decision ─────────────────────────────────────────────────────
            APPROVE_T = 531.8
            REVIEW_T  = 510.4
            if score >= APPROVE_T:
                decision = "approve"
            elif score >= REVIEW_T:
                decision = "review"
            else:
                decision = "reject"

            # ── Result display ────────────────────────────────────────────────
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown('<p class="section-title">Scoring Result</p>', unsafe_allow_html=True)

            r1, r2, r3 = st.columns([1, 1, 2], gap="medium")

            with r1:
                st.plotly_chart(score_gauge(score), use_container_width=True,
                                config={"displayModeBar": False})

            with r2:
                dec_label = {"approve": "APPROVE", "review": "REVIEW", "reject": "REJECT"}[decision]
                st.markdown(f"""
<div class="result-card {decision}">
  <p class="result-decision {decision}">{dec_label}</p>
  <p class="result-score">{score:.1f}</p>
  <p class="result-pd">PD: {pd_prob*100:.2f}%</p>
</div>""", unsafe_allow_html=True)

            with r3:
                st.markdown('<p class="section-title" style="margin-top:8px">WOE Contributions</p>', unsafe_allow_html=True)
                live_woe = [
                    ("total_delinquency",  woe_td),
                    ("revolving_util",     woe_util),
                    ("90days_late",        woe_90),
                    ("age",                woe_age),
                    ("income_per_person",  woe_inc),
                ]
                bars = "".join(_woe_bar(f, v) for f, v in live_woe)
                st.markdown(
                    f'<div style="background:#1e2019;border:0.5px solid #2e2e28;'
                    f'border-radius:12px;padding:16px 18px">{bars}</div>',
                    unsafe_allow_html=True,
                )

            # ── Summary stats row ─────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4, gap="small")
            with s1: st.markdown(metric_card("Credit score",      f"{score:.1f}",        "good" if score >= APPROVE_T else "warn" if score >= REVIEW_T else "bad", f"Range: 366–551"), unsafe_allow_html=True)
            with s2: st.markdown(metric_card("PD probability",    f"{pd_prob*100:.2f}%", "bad" if pd_prob > 0.15 else "warn" if pd_prob > 0.05 else "good",       "Prob. of default"), unsafe_allow_html=True)
            with s3: st.markdown(metric_card("Total delinquency", str(total_delinquency),"bad" if total_delinquency > 2 else "warn" if total_delinquency > 0 else "good", "Sum of all late events"), unsafe_allow_html=True)
            with s4: st.markdown(metric_card("Income / person",   f"${income_per_person:,.0f}", "good" if income_per_person > 3000 else "warn" if income_per_person > 1500 else "bad", "Monthly income per capita"), unsafe_allow_html=True)

            # ── API block (shown but non-blocking if not running) ─────────────
            with st.expander("API integration (optional)", expanded=False):
                st.markdown(
                    '<p style="font-size:12px;color:#5a5a50;margin-bottom:12px">'
                    'Connect to a live API endpoint to override the in-browser scoring above.</p>',
                    unsafe_allow_html=True,
                )
                api_url = st.text_input("API endpoint", value="http://localhost:8000/api/v1/predict")
                if st.button("Call API"):
                    import requests as req
                    payload = {
                        "features": {
                            "RevolvingUtilizationOfUnsecuredLines": revolving_util,
                            "age": age,
                            "NumberOfTime30-59DaysPastDueNotWorse": times_30_59,
                            "DebtRatio": debt_ratio,
                            "MonthlyIncome": monthly_income,
                            "NumberOfOpenCreditLinesAndLoans": open_credit,
                            "NumberOfTimes90DaysLate": times_90,
                            "NumberRealEstateLoansOrLines": real_estate,
                            "NumberOfTime60-89DaysPastDueNotWorse": times_60_89,
                            "NumberOfDependents": num_dependents,
                        }
                    }
                    try:
                        resp = req.post(api_url, json=payload, timeout=5)
                        if resp.status_code == 200:
                            st.json(resp.json())
                        else:
                            st.error(f"API returned {resp.status_code}: {resp.text}")
                    except Exception as e:
                        st.warning(f"API not reachable: {e}")


# =============================================================================
# PAGES: PROJECTS 2–5  (under construction)
# =============================================================================

else:
    titles = {
        "2 · Loan Amount Prediction":    ("Loan Amount Prediction",    "Regression model estimating the appropriate loan amount for a borrower."),
        "3 · Early Payment Prediction":  ("Early Payment Prediction",  "Forecast prepayment risk and expected loan duration."),
        "4 · Credit Score Prediction":   ("Credit Score Prediction",   "AI-driven credit scoring using alternative data sources."),
        "5 · Loan Recovery Prediction":  ("Loan Recovery Prediction",  "Estimate recovery amounts for defaulted loans."),
    }
    title, desc = titles.get(page, ("Module", ""))
    st.markdown(f'<p class="page-title">{title}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-subtitle">{desc}</p>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="wip-card">
  <p class="wip-label">Status</p>
  <p class="wip-title">Module under construction 🚧</p>
  <p class="wip-sub">This project is planned for a future release.<br>
  Use the sidebar to navigate to the live Credit Default Prediction module.</p>
</div>""", unsafe_allow_html=True)