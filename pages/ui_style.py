"""
ui_style.py
===========
Shared style helper for the Portfolio Accounting Engine Streamlit app.
Embeds the CSS design system from the reference site (index.html) and
injects it into Streamlit via st.markdown(unsafe_allow_html=True).

Usage:
    from ui_style import inject_site_css, render_hero, section_sep, ...
"""

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CSS — extracted and adapted from index.html for Streamlit
# ─────────────────────────────────────────────────────────────────────────────

_SITE_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,300&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0b0e14;
    --surface: #12161f;
    --surface2: #1a1f2e;
    --border: #232a3a;
    --accent: #4fffb0;
    --accent2: #7b8cff;
    --accent3: #ff8c61;
    --text: #d4dbe8;
    --text-muted: #6b7794;
    --text-dim: #3d4866;
    --heading: #eef2ff;
}

/* ── Streamlit App Background ── */
.stApp, [data-testid="stAppViewContainer"], .main .block-container {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 300 !important;
}

section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stDateInput label {
    color: var(--text-muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}

/* ── Headers ── */
h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
    color: var(--heading) !important;
}

h1 { font-weight: 700 !important; }
h2 { font-weight: 600 !important; }
h3 { font-weight: 600 !important; }

/* ── Streamlit metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem 1.25rem !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] {
    color: var(--heading) !important;
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
}

[data-testid="stMetricDelta"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"], .stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: var(--bg) !important;
    border: none !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
}

.stButton > button[kind="primary"]:hover {
    background: #3de09a !important;
}

/* ── Tabs (for pages) ── */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    color: var(--text-muted) !important;
    padding: 0.75rem 1.25rem !important;
}

.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    background: var(--surface) !important;
}

[data-testid="stExpander"] summary {
    color: var(--text-muted) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Horizontal rule ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
}

/* ── Line chart background fix ── */
[data-testid="stVegaLiteChart"] {
    border-radius: 8px !important;
}

/* ── Toast/info/warning blocks ── */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

/* ════════════════════════════════════════════════════════════ */
/* CUSTOM COMPONENT CLASSES (used in st.markdown HTML blocks) */
/* ════════════════════════════════════════════════════════════ */

.vise-hero {
    padding: 2.5rem 0 2rem;
    position: relative;
}

.vise-hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.vise-hero-title {
    font-family: 'Fraunces', serif;
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 700;
    color: var(--heading);
    line-height: 1.1;
    margin-bottom: 1rem;
}

.vise-hero-title em {
    font-style: italic;
    font-weight: 300;
    color: var(--accent);
}

.vise-hero-sub {
    font-size: 1rem;
    color: var(--text-muted);
    max-width: 560px;
    line-height: 1.7;
    margin-bottom: 1.5rem;
}

.vise-formula {
    display: inline-flex;
    align-items: center;
    gap: 0.75rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    padding: 0.85rem 1.25rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: var(--heading);
    border-radius: 4px;
}

.vise-formula span { color: var(--accent2); }

/* ── Section Separator ── */
.vise-sep {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    margin: 2.5rem 0 1.5rem;
}

.vise-sep-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}

.vise-sep-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-dim);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    white-space: nowrap;
}

/* ── Section Label ── */
.vise-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--accent);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Section Title ── */
.vise-section-title {
    font-family: 'Fraunces', serif;
    font-size: clamp(1.6rem, 3vw, 2.4rem);
    font-weight: 600;
    color: var(--heading);
    line-height: 1.2;
    margin-bottom: 1rem;
}

/* ── Section Intro ── */
.vise-section-intro {
    font-size: 1rem;
    color: var(--text-muted);
    max-width: 640px;
    margin-bottom: 2rem;
    line-height: 1.75;
}

/* ── Architecture / Component Cards ── */
.vise-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 2rem;
}

.vise-card {
    background: var(--surface);
    padding: 1.75rem;
    transition: background 0.2s;
}

.vise-card:hover { background: var(--surface2); }

.vise-card-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-dim);
    letter-spacing: 0.15em;
    margin-bottom: 0.6rem;
}

.vise-card-title {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--heading);
    margin-bottom: 0.4rem;
}

.vise-card-tag {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    padding: 0.2rem 0.6rem;
    border-radius: 3px;
    margin-bottom: 0.6rem;
}

.tag-stateful { background: rgba(79,255,176,0.1); color: var(--accent); }
.tag-stateless { background: rgba(123,140,255,0.1); color: var(--accent2); }
.tag-orchestrator { background: rgba(255,140,97,0.1); color: var(--accent3); }

.vise-card p {
    font-size: 0.86rem;
    color: var(--text-muted);
    line-height: 1.6;
    margin: 0;
}

/* ── Tax Cards ── */
.vise-tax-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.25rem;
    margin-bottom: 2rem;
}

.vise-tax-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}

.vise-tax-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}

.vise-tax-card.st::before { background: var(--accent3); }
.vise-tax-card.lt::before { background: var(--accent); }
.vise-tax-card.div::before { background: var(--accent2); }

.vise-tax-rate {
    font-family: 'Fraunces', serif;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.35rem;
}

.vise-tax-card.st .vise-tax-rate { color: var(--accent3); }
.vise-tax-card.lt .vise-tax-rate { color: var(--accent); }
.vise-tax-card.div .vise-tax-rate { color: var(--accent2); }

.vise-tax-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.vise-tax-desc {
    font-size: 0.84rem;
    color: var(--text-muted);
    line-height: 1.6;
}

/* ── Principle Pills ── */
.vise-principle {
    display: flex;
    gap: 0.9rem;
    align-items: flex-start;
    padding: 1rem 1.1rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}

.vise-principle:hover { border-color: var(--accent); }

.vise-principle-icon {
    flex-shrink: 0;
    width: 28px; height: 28px;
    border-radius: 50%;
    background: rgba(79,255,176,0.1);
    border: 1px solid rgba(79,255,176,0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    color: var(--accent);
    font-family: 'DM Mono', monospace;
    margin-top: 0.1rem;
}

.vise-principle-title {
    font-weight: 500;
    color: var(--heading);
    font-size: 0.88rem;
    margin-bottom: 0.15rem;
}

.vise-principle-body {
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.6;
}

/* ── Steps (vertical timeline) ── */
.vise-steps {
    position: relative;
    padding-left: 2.5rem;
}

.vise-steps::before {
    content: '';
    position: absolute;
    left: 14px; top: 20px; bottom: 20px;
    width: 1px;
    background: linear-gradient(to bottom, var(--accent), var(--accent2), transparent);
    opacity: 0.3;
}

.vise-step {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    padding: 0.9rem 0;
    position: relative;
}

.vise-step-num {
    position: absolute;
    left: -2.5rem;
    width: 30px; height: 30px;
    border-radius: 50%;
    background: var(--surface2);
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent);
    z-index: 1;
}

.vise-step-title {
    font-weight: 500;
    color: var(--heading);
    font-size: 0.9rem;
    margin-bottom: 0.2rem;
}

.vise-step-detail {
    font-family: 'DM Mono', monospace;
    font-size: 0.76rem;
    color: var(--text-muted);
    line-height: 1.6;
}

/* ── Code Block ── */
.vise-code {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.1rem 1.25rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    line-height: 1.7;
    overflow-x: auto;
    margin-bottom: 1rem;
    white-space: pre-wrap;
}

.vise-code .kw { color: var(--accent2); }
.vise-code .s { color: var(--accent); }
.vise-code .cm { color: var(--text-dim); }
.vise-code .fn { color: var(--accent3); }

/* ── Guardrails ── */
.vise-guardrails {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 2rem;
}

.vise-guardrail {
    background: var(--surface);
    padding: 1.25rem;
    transition: background 0.2s;
}

.vise-guardrail:hover { background: var(--surface2); }

.vise-guardrail-trigger {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent3);
    margin-bottom: 0.4rem;
}

.vise-guardrail-response {
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.55;
}

.vise-guardrail-response strong { color: var(--accent); font-weight: 500; }

/* ── Class Reference ── */
.vise-classref {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 6px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 1.25rem;
}

.vise-classref-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.76rem;
    color: var(--accent2);
    margin-bottom: 0.6rem;
    letter-spacing: 0.05em;
}

.vise-classref-method {
    display: flex;
    gap: 0.6rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.78rem;
    align-items: flex-start;
    flex-wrap: wrap;
}

.vise-classref-method:last-child { border-bottom: none; }

.vise-crm-sig {
    font-family: 'DM Mono', monospace;
    color: var(--accent2);
    font-size: 0.72rem;
    flex-shrink: 0;
    min-width: 180px;
}

.vise-crm-desc {
    color: var(--text-muted);
    line-height: 1.5;
    font-size: 0.8rem;
}

/* ── Outputs List ── */
.vise-outputs {
    display: flex;
    flex-direction: column;
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 2rem;
}

.vise-output-item {
    display: grid;
    grid-template-columns: 200px 1fr;
    background: var(--surface);
    transition: background 0.2s;
}

.vise-output-item:hover { background: var(--surface2); }

.vise-output-name {
    padding: 1.1rem 1.25rem;
    border-right: 1px solid var(--border);
    font-family: 'Fraunces', serif;
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--heading);
    display: flex;
    align-items: center;
}

.vise-output-desc {
    padding: 1.1rem 1.25rem;
    font-size: 0.84rem;
    color: var(--text-muted);
    line-height: 1.6;
    display: flex;
    align-items: center;
}

/* ── Config Table ── */
.vise-config-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 1rem;
}

.vise-config-header {
    padding: 0.85rem 1.25rem;
    border-bottom: 1px solid var(--border);
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.vise-config-row {
    padding: 0.7rem 1.25rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.8rem;
    display: flex;
    gap: 1rem;
    align-items: baseline;
}

.vise-config-row:last-child { border-bottom: none; }

.vise-config-key {
    font-family: 'DM Mono', monospace;
    color: var(--accent2);
    font-size: 0.76rem;
    min-width: 180px;
}

.vise-config-val {
    font-family: 'DM Mono', monospace;
    color: var(--accent);
    font-size: 0.78rem;
    min-width: 60px;
}

.vise-config-desc {
    color: var(--text-muted);
    font-size: 0.78rem;
}

/* ── Two Column ── */
.vise-two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-bottom: 2rem;
}

@media (max-width: 768px) {
    .vise-two-col { grid-template-columns: 1fr; }
    .vise-output-item { grid-template-columns: 1fr; }
    .vise-output-name { border-right: none; border-bottom: 1px solid var(--border); }
}

/* ── Footer ── */
.vise-footer {
    border-top: 1px solid var(--border);
    padding: 1.5rem 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 3rem;
}

.vise-footer-brand {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-dim);
    letter-spacing: 0.1em;
}

/* ── Inline code override ── */
.vise-code-inline {
    font-family: 'DM Mono', monospace;
    font-size: 0.8em;
    background: var(--surface2);
    border: 1px solid var(--border);
    padding: 0.1em 0.4em;
    border-radius: 3px;
    color: var(--accent2);
}

/* ── Worked Example ── */
.vise-example-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 2rem;
}

.vise-example-header {
    display: grid;
    grid-template-columns: 1fr 1fr;
    border-bottom: 1px solid var(--border);
}

.vise-example-col {
    padding: 1.25rem 1.5rem;
}

.vise-example-col:first-child {
    border-right: 1px solid var(--border);
}

.vise-example-col-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.64rem;
    color: var(--text-dim);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

.vise-example-line {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.3rem 0;
    font-size: 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}

.vise-example-line:last-child { border-bottom: none; }
.vise-example-line .label { color: var(--text-muted); }
.vise-example-line .value { font-family: 'DM Mono', monospace; color: var(--heading); }
.vise-example-line .value.pos { color: var(--accent); }
.vise-example-line .value.neg { color: var(--accent3); }

.vise-example-result {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    border-top: 1px solid var(--border);
}

.vise-result-item {
    padding: 1.25rem 1.5rem;
    border-right: 1px solid var(--border);
}

.vise-result-item:last-child { border-right: none; }

.vise-result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.64rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.vise-result-value {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--heading);
}

.vise-result-value.accent { color: var(--accent); }
.vise-result-value.warn { color: var(--accent3); }

/* ── Formula Visual ── */
.vise-formula-visual {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
}

.vise-formula-big {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--heading);
    line-height: 1.3;
    margin-bottom: 0.75rem;
}

.vise-formula-big em { color: var(--accent); font-style: italic; }
.vise-formula-big span { color: var(--accent2); }

/* ── Sidebar Navigation Enhance ── */
section[data-testid="stSidebar"] h1 {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--accent) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}

/* ── Multipage sidebar links ── */
[data-testid="stSidebarNav"] a {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-muted) !important;
    font-size: 0.88rem !important;
}

[data-testid="stSidebarNav"] a:hover {
    color: var(--accent) !important;
}

[data-testid="stSidebarNav"] [aria-current="page"] {
    background: var(--surface2) !important;
    border-left: 2px solid var(--accent) !important;
}

[data-testid="stSidebarNav"] [aria-current="page"] a {
    color: var(--accent) !important;
}
</style>
"""


def inject_site_css():
    """Inject the full VISE design system CSS into a Streamlit page."""
    st.markdown(_SITE_CSS, unsafe_allow_html=True)


def render_hero(eyebrow: str, title: str, subtitle: str = "", formula: str = ""):
    """Render a hero block matching the reference site."""
    html = f'<div class="vise-hero">'
    html += f'<div class="vise-hero-eyebrow">{eyebrow}</div>'
    html += f'<div class="vise-hero-title">{title}</div>'
    if subtitle:
        html += f'<div class="vise-hero-sub">{subtitle}</div>'
    if formula:
        html += f'<div class="vise-formula">{formula}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def section_sep(number: str, label: str):
    """Render a section separator with number and label."""
    st.markdown(
        f'<div class="vise-sep">'
        f'<div class="vise-sep-line"></div>'
        f'<div class="vise-sep-label">{number} — {label}</div>'
        f'<div class="vise-sep-line"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(label: str, title: str, intro: str = ""):
    """Render label + title + intro paragraph."""
    html = f'<div class="vise-label">{label}</div>'
    html += f'<div class="vise-section-title">{title}</div>'
    if intro:
        html += f'<div class="vise-section-intro">{intro}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_footer():
    """Render a site footer."""
    st.markdown(
        '<div class="vise-footer">'
        '<div class="vise-footer-brand">PORTFOLIO ACCOUNTING ENGINE</div>'
        '<div class="vise-footer-brand">UTexas MSBA // VISE</div>'
        '</div>',
        unsafe_allow_html=True,
    )
