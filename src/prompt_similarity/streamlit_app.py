"""
app.py — Streamlit Frontend for Prompt Similarity Service
Run: streamlit run app.py
Requires: pip install streamlit httpx plotly numpy
"""

import json
import httpx
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prompt Similarity",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_URL = "http://localhost:8000"

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0b0e !important;
    color: #e8e6e0 !important;
    font-family: 'Space Mono', monospace !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }

/* ── Main layout ── */
.main .block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1280px !important;
}

/* ── Hero Header ── */
.hero-header {
    display: flex;
    align-items: flex-end;
    gap: 1.5rem;
    padding: 2.5rem 0 2rem;
    border-bottom: 1px solid #1e2028;
    margin-bottom: 2rem;
}
.hero-glyph {
    font-size: 3.2rem;
    line-height: 1;
    color: #c8f55a;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
}
.hero-text h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #f0ede6;
    margin: 0 0 0.15rem;
}
.hero-text p {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #5a5e6b;
    margin: 0;
}
.health-badge {
    margin-left: auto;
    padding: 0.45rem 1rem;
    border-radius: 2px;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    text-transform: uppercase;
}
.health-ok   { background: #1a2a0a; color: #c8f55a; border: 1px solid #c8f55a22; }
.health-err  { background: #2a0a0a; color: #f55a5a; border: 1px solid #f55a5a22; }
.health-num  { font-size: 0.62rem; color: #5a5e6b; margin-left: 1rem; }

/* ── Tab navigation ── */
[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid #1e2028;
    gap: 0 !important;
}
button[data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #3e4252 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 0 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}
button[data-baseweb="tab"]:hover { color: #9da0ac !important; }
button[aria-selected="true"][data-baseweb="tab"] {
    color: #c8f55a !important;
    border-bottom-color: #c8f55a !important;
}
[data-testid="stTabPanel"] { padding-top: 2rem !important; }

/* ── Section labels ── */
.section-label {
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3e4252;
    margin-bottom: 0.75rem;
    padding-left: 0.1rem;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
    background: #111318 !important;
    border: 1px solid #1e2028 !important;
    border-radius: 2px !important;
    color: #e8e6e0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #c8f55a44 !important;
    box-shadow: 0 0 0 2px #c8f55a11 !important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #5a5e6b !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-testid="stMarkdown"] p {
    font-size: 0.7rem !important; color: #5a5e6b !important;
}
[class*="StyledThumb"] { background: #c8f55a !important; border-color: #c8f55a !important; }
[class*="StyledTrack"][aria-valuenow] { background: #c8f55a !important; }

/* ── Buttons ── */
[data-testid="stButton"] button {
    background: #c8f55a !important;
    color: #0a0b0e !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.6rem !important;
    transition: opacity 0.2s !important;
}
[data-testid="stButton"] button:hover { opacity: 0.85 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #111318 !important;
    border: 1px dashed #2a2d38 !important;
    border-radius: 2px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #5a5e6b !important;
}

/* ── Result Cards ── */
.result-card {
    background: #111318;
    border: 1px solid #1e2028;
    border-left: 3px solid #c8f55a;
    border-radius: 2px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.result-card:hover { border-left-color: #deff6e; }

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.card-id {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    color: #c8f55a;
    letter-spacing: 0.04em;
}
.card-score {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #5a5e6b;
    background: #1a1d24;
    padding: 0.2rem 0.5rem;
    border-radius: 2px;
    border: 1px solid #2a2d38;
}
.card-preview {
    font-size: 0.75rem;
    color: #7a7e8c;
    line-height: 1.55;
    word-break: break-word;
}

/* ── Cluster Cards ── */
.cluster-card {
    background: #111318;
    border: 1px solid #1e2028;
    border-radius: 2px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.cluster-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #1e2028;
}
.cluster-badge {
    background: #c8f55a;
    color: #0a0b0e;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 0.2rem 0.55rem;
    border-radius: 2px;
}
.cluster-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    color: #9da0ac;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.cluster-prompt-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid #15171e;
    font-size: 0.72rem;
}
.cluster-prompt-row:last-child { border-bottom: none; }
.cluster-prompt-id { color: #c8f55a; font-family: 'Space Mono', monospace; }
.cluster-sim { color: #5a5e6b; }

.merge-box {
    margin-top: 0.9rem;
    background: #0d1219;
    border: 1px solid #1e2028;
    border-left: 3px solid #3a5e9a;
    border-radius: 2px;
    padding: 0.75rem 1rem;
}
.merge-box-label {
    font-size: 0.6rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #3a5e9a;
    margin-bottom: 0.35rem;
}
.merge-box-note {
    font-size: 0.72rem;
    color: #6a7080;
    line-height: 1.5;
}
.merge-var {
    display: inline-block;
    background: #1a2535;
    color: #7ab0f5;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 0.15rem 0.4rem;
    border-radius: 2px;
    margin: 0.15rem 0.15rem 0 0;
    border: 1px solid #2a3a55;
}

/* ── Latency Bar ── */
.latency-bar {
    display: inline-flex;
    gap: 1.25rem;
    align-items: center;
    background: #0d0f14;
    border: 1px solid #1e2028;
    border-radius: 2px;
    padding: 0.45rem 1rem;
    margin-top: 1.5rem;
}
.latency-item {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
}
.latency-label {
    font-size: 0.55rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #3e4252;
    line-height: 1;
}
.latency-value {
    font-size: 0.8rem;
    font-weight: 700;
    color: #c8f55a;
    font-family: 'Space Mono', monospace;
    line-height: 1.4;
}
.latency-sep { width: 1px; height: 28px; background: #1e2028; }

/* ── JSON block ── */
.json-block {
    background: #080a0d;
    border: 1px solid #1e2028;
    border-radius: 2px;
    padding: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #6a7080;
    overflow-x: auto;
    white-space: pre;
    max-height: 280px;
    overflow-y: auto;
}

/* ── Template / info boxes ── */
.info-box {
    background: #0d1219;
    border: 1px solid #1e2028;
    border-left: 3px solid #3a5e9a;
    border-radius: 2px;
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.72rem;
    color: #6a7080;
    line-height: 1.6;
}

/* ── Divider ── */
[data-testid="stMarkdownContainer"] hr {
    border: none;
    border-top: 1px solid #1e2028;
    margin: 1.5rem 0;
}

/* ── st.code override ── */
[data-testid="stCode"] {
    background: #080a0d !important;
    border: 1px solid #1e2028 !important;
    border-radius: 2px !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: #1a2a0a !important;
    border: 1px solid #c8f55a22 !important;
    border-radius: 2px !important;
    color: #c8f55a !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
}
[data-testid="stException"] {
    background: #2a0a0a !important;
    border: 1px solid #f55a5a22 !important;
    border-radius: 2px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #111318 !important;
    border: 1px solid #1e2028 !important;
    border-radius: 2px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #5a5e6b !important;
}

/* ── Column gaps ── */
[data-testid="column"] { padding: 0 0.5rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0a0b0e; }
::-webkit-scrollbar-thumb { background: #2a2d38; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs):
    try:
        r = getattr(httpx, method)(f"{BASE_URL}{path}", timeout=120, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except httpx.ConnectError:
        return None, "Cannot connect to server. Is `uvicorn main:app --reload` running?"
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("detail", e.response.text)
        except Exception:
            detail = e.response.text
        return None, f"HTTP {e.response.status_code}: {detail}"
    except Exception as e:
        return None, str(e)


def latency_html(items: dict) -> str:
    parts = []
    for label, val in items.items():
        parts.append(f"""
        <div class="latency-item">
            <span class="latency-label">{label}</span>
            <span class="latency-value">{val}</span>
        </div>""")
    inner = '<div class="latency-sep"></div>'.join(parts)
    return f'<div class="latency-bar">{inner}</div>'


def result_card(prompt_id: str, score: float, preview: str) -> str:
    bar_w = int(score * 100)
    bar_color = "#c8f55a" if score >= 0.8 else "#f5b55a" if score >= 0.6 else "#5a7ef5"
    return f"""
    <div class="result-card">
        <div class="card-header">
            <span class="card-id">{prompt_id}</span>
            <span class="card-score" style="color:{bar_color}">
                {score:.4f}
                <span style="color:#2a2d38;margin:0 0.3rem">|</span>
                <span style="display:inline-block;width:{bar_w}px;max-width:80px;height:2px;
                      background:{bar_color};vertical-align:middle;border-radius:1px"></span>
            </span>
        </div>
        <div class="card-preview">{preview}</div>
    </div>"""


def render_cluster_card(cluster: dict):
    """Render a cluster card using native Streamlit components."""
    cid     = cluster["cluster_id"]
    prompts = cluster["prompts"]
    merge   = cluster["merge_suggestion"]

    with st.container(border=True):
        # Header row
        st.markdown(f"**Cluster {cid}** &nbsp; `{len(prompts)} prompts`")
        st.divider()

        # Prompt rows: id on the left, similarity on the right
        for i, p in enumerate(prompts):
            sim_disp = "★ anchor" if i == 0 else f"{p['similarity']:.4f}"
            col_id, col_sim = st.columns([3, 1])
            with col_id:
                st.code(p["prompt_id"], language=None)
            with col_sim:
                st.caption(sim_disp)

        # Merge suggestion
        st.divider()
        vars_list = merge.get("unified_variables", [])
        vars_str  = "  ".join(f"`{{{{{v}}}}}`" for v in vars_list) if vars_list else "_none_"
        st.markdown(f"**⟡ Merge suggestion**  \n{merge['note']}")
        if vars_list:
            st.markdown(f"Variables: {vars_str}")


def scatter_clusters(clusters: list) -> go.Figure:
    """
    2-D grid scatter. Clusters are placed on a sqrt-balanced grid so 300
    clusters don't collapse onto a single axis. Each cluster's points are
    jittered inside its cell. Connecting lines radiate from the representative.
    Text labels are suppressed above 40 clusters (hover only).
    """
    rng = np.random.default_rng(42)
    fig = go.Figure()

    n_clusters = len(clusters)

    # ── Grid geometry ─────────────────────────────────────────────────────────
    grid_cols  = max(1, int(np.ceil(np.sqrt(n_clusters * 1.6))))  # wider than tall
    cell_size  = 3.0          # spacing between cluster centres
    jitter     = cell_size * 0.38   # max spread within a cell

    # ── Density-aware rendering ───────────────────────────────────────────────
    show_text   = n_clusters <= 40
    marker_size = max(6, 14 - int(n_clusters / 25))   # shrink dots at scale
    line_width  = 1 if n_clusters <= 80 else 0.5

    palette = [
        "#c8f55a", "#5ab4f5", "#f5855a", "#a55af5",
        "#f5d05a", "#5af5c8", "#f55a9a", "#f57a5a",
        "#5af5a0", "#f5a05a", "#5a8af5", "#d45af5",
        "#f5e05a", "#5af5e0", "#f55a7a", "#8af55a",
    ]

    for idx, c in enumerate(clusters):
        cid     = c["cluster_id"]
        prompts = c["prompts"]
        n       = len(prompts)
        color   = palette[idx % len(palette)]

        # Grid cell centre
        grid_row = idx // grid_cols
        grid_col = idx  % grid_cols
        cx = grid_col * cell_size
        cy = grid_row * cell_size

        # Distribute points within the cell (polar jitter for even spread)
        angles = rng.uniform(0, 2 * np.pi, n)
        radii  = rng.uniform(0, jitter, n)
        xs = cx + radii * np.cos(angles)
        ys = cy + radii * np.sin(angles)

        ids    = [p["prompt_id"] for p in prompts]
        sims   = [p["similarity"] for p in prompts]
        hovers = [
            f"<b>Cluster {cid}</b><br>{pid}<br>sim: {s:.4f}"
            for pid, s in zip(ids, sims)
        ]

        text_labels = [pid.split("_")[-1][:10] for pid in ids] if show_text else [""] * n

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text" if show_text else "markers",
            text=text_labels,
            textposition="top center",
            textfont=dict(size=8, color=color, family="Space Mono"),
            hovertext=hovers,
            hoverinfo="text",
            marker=dict(
                size=marker_size,
                color=color,
                opacity=0.80,
                line=dict(color="#0a0b0e", width=1.5),
            ),
            name=f"C{cid}",
            legendgroup=str(cid),
            showlegend=n_clusters <= 60,
        ))

        # ── Cluster label annotation (centroid) ───────────────────────────────
        if not show_text:
            fig.add_annotation(
                x=cx, y=cy,
                text=f"<b>C{cid}</b>",
                showarrow=False,
                font=dict(size=7, color=color, family="Space Mono"),
                opacity=0.6,
                xanchor="center", yanchor="middle",
            )

        # ── Connecting lines from representative (index 0) ────────────────────
        if n > 1:
            rep_x, rep_y = xs[0], ys[0]
            for j in range(1, n):
                fig.add_shape(
                    type="line",
                    x0=rep_x, y0=rep_y, x1=xs[j], y1=ys[j],
                    line=dict(color=color, width=line_width, dash="dot"),
                    opacity=0.25,
                )

    # ── Dynamic height: enough rows × px per row ──────────────────────────────
    n_rows  = max(1, int(np.ceil(n_clusters / grid_cols)))
    px_row  = 120 if n_clusters > 100 else 160
    height  = max(500, min(1400, n_rows * px_row + 80))

    fig.update_layout(
        paper_bgcolor="#0a0b0e",
        plot_bgcolor="#0d0f14",
        font=dict(family="Space Mono", color="#5a5e6b", size=10),
        xaxis=dict(
            showgrid=True, gridcolor="#15171e", zeroline=False,
            showticklabels=False, showline=False,
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#15171e", zeroline=False,
            showticklabels=False, showline=False,
            scaleanchor="x", scaleratio=1,   # square cells
        ),
        legend=dict(
            bgcolor="#111318", bordercolor="#1e2028", borderwidth=1,
            font=dict(size=8, color="#9da0ac"),
            itemsizing="constant",
            tracegroupgap=2,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
        hoverlabel=dict(
            bgcolor="#111318", bordercolor="#2a2d38",
            font=dict(family="Space Mono", size=11, color="#e8e6e0"),
            align="left",
        ),
        dragmode="pan",
    )
    return fig


# ── Health Check ───────────────────────────────────────────────────────────────
health_data, health_err = api("get", "/health")

# ── Header ─────────────────────────────────────────────────────────────────────
h_left, h_right = st.columns([3, 2], gap="large")
with h_left:
    st.markdown("## ⬡ Prompt Similarity")
    st.caption("Embedding · Search · Deduplication")
with h_right:
    if health_data:
        indexed = health_data.get("prompts_indexed", 0)
        mb      = health_data.get("cache_memory_mb", 0)
        model   = health_data.get("model", "—")
        st.success(f"Live  —  {indexed} indexed  ·  {mb} MB  ·  {model}")
    else:
        st.error("Offline — server not reachable")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "01 · Generate Embeddings",
    "02 · Find Similar",
    "03 · Semantic Search",
    "04 · Duplicate Clusters",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Generate Embeddings
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        st.markdown('<div class="section-label">Upload JSON file</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            JSON must be an array of prompt objects. Each object requires
            <code>prompt_id</code>, <code>category</code>, <code>layer</code>,
            and <code>content</code>. Optional: <code>name</code>.
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("prompts.json", type=["json"], label_visibility="collapsed")

        if uploaded:
            try:
                data = json.loads(uploaded.read())
                st.markdown(f'<div class="section-label">{len(data)} prompts loaded</div>',
                            unsafe_allow_html=True)
                with st.expander("Preview JSON"):
                    st.markdown(
                        f'<div class="json-block">{json.dumps(data[:3], indent=2)}</div>',
                        unsafe_allow_html=True,
                    )
                if st.button("⟢  Generate Embeddings", key="gen_file"):
                    with st.spinner("Calling embedding API…"):
                        res, err = api("post", "/api/embeddings/generate", json=data)
                    if err:
                        st.error(err)
                    else:
                        st.success(f"✓ Generated {res['generated']} embeddings.")
                        lat = res.get("latency", {})
                        st.markdown(latency_html({
                            "Embed": f"{lat.get('embed_ms', '—')} ms",
                            "Cache rebuild": f"{lat.get('cache_rebuild_ms', '—')} ms",
                            "Per text": f"{lat.get('ms_per_text', '—')} ms",
                            "Total indexed": str(res.get('cache', {}).get('total_vectors', '—')),
                        }), unsafe_allow_html=True)
            except json.JSONDecodeError:
                st.error("Invalid JSON file.")

    with col_b:
        st.markdown('<div class="section-label">Manual entry</div>', unsafe_allow_html=True)

        TEMPLATE = json.dumps([{
            "prompt_id": "greeting_001",
            "category":  "greeting",
            "layer":     "system",
            "name":      "Basic Greeting",
            "content":   "Hello, I'm {{agent_name}} from {{org_name}}. How can I help?"
        }], indent=2)

        manual_json = st.text_area(
            "Prompt array (JSON)",
            value=TEMPLATE,
            height=260,
            label_visibility="collapsed",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("⟢  Submit Prompts", key="gen_manual"):
                try:
                    prompts = json.loads(manual_json)
                    if not isinstance(prompts, list):
                        st.error("Must be a JSON array.")
                    else:
                        with st.spinner("Generating…"):
                            res, err = api("post", "/api/embeddings/generate", json=prompts)
                        if err:
                            st.error(err)
                        else:
                            st.success(f"✓ {res['generated']} embeddings generated.")
                            lat = res.get("latency", {})
                            st.markdown(latency_html({
                                "Embed": f"{lat.get('embed_ms', '—')} ms",
                                "Cache": f"{lat.get('cache_rebuild_ms', '—')} ms",
                                "Indexed": str(res.get('cache', {}).get('total_vectors', '—')),
                            }), unsafe_allow_html=True)
                except json.JSONDecodeError as e:
                    st.error(f"JSON parse error: {e}")

        with c2:
            if st.button("↻  Re-embed All", key="gen_all"):
                with st.spinner("Re-embedding all prompts…"):
                    res, err = api("post", "/api/embeddings/generate",
                                   params={"regenerate_all": True})
                if err:
                    st.error(err)
                else:
                    st.success(f"✓ Re-embedded {res['generated']} prompts.")
                    lat = res.get("latency", {})
                    st.markdown(latency_html({
                        "Embed": f"{lat.get('embed_ms', '—')} ms",
                        "Cache": f"{lat.get('cache_rebuild_ms', '—')} ms",
                    }), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Find Similar
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="section-label">Parameters</div>', unsafe_allow_html=True)
        prompt_id = st.text_input("Prompt ID", placeholder="e.g. greeting_001")
        sim_limit = st.number_input("Limit", min_value=1, max_value=50, value=5)
        sim_thresh = st.slider("Similarity threshold", 0.0, 1.0, 0.8, 0.01,
                               format="%.2f", key="sim_thresh")
        find_btn = st.button("⟢  Find Similar", key="find_sim")

    with col_r:
        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
        if find_btn:
            if not prompt_id.strip():
                st.error("Enter a prompt ID.")
            else:
                with st.spinner("Searching…"):
                    res, err = api("get",
                                   f"/api/prompts/{prompt_id.strip()}/similar",
                                   params={"limit": sim_limit, "threshold": sim_thresh})
                if err:
                    st.error(err)
                else:
                    results = res.get("results", [])
                    if not results:
                        st.markdown(
                            '<div class="info-box">No results above threshold.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        for r in results:
                            st.markdown(
                                result_card(r["prompt_id"], r["similarity_score"],
                                            r["content_preview"]),
                                unsafe_allow_html=True,
                            )
                    lat = res.get("latency", {})
                    st.markdown(latency_html({
                        "Search": f"{lat.get('search_ms', '—')} ms",
                        "Hits": str(len(results)),
                    }), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Semantic Search
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="section-label">Query</div>', unsafe_allow_html=True)
        query = st.text_area("Natural-language query",
                             placeholder="e.g. Ask the patient for their date of birth",
                             height=110, label_visibility="collapsed")
        ss_limit = st.number_input("Limit", min_value=1, max_value=50, value=10,
                                   key="ss_limit")
        ss_thresh = st.slider("Min similarity", 0.0, 1.0, 0.0, 0.01,
                              format="%.2f", key="ss_thresh")
        search_btn = st.button("⟢  Search", key="sem_search")

    with col_r:
        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
        if search_btn:
            if not query.strip():
                st.error("Enter a search query.")
            else:
                with st.spinner("Embedding & searching…"):
                    res, err = api("post", "/api/search/semantic", json={
                        "query": query.strip(),
                        "limit": ss_limit,
                        "threshold": ss_thresh,
                    })
                if err:
                    st.error(err)
                else:
                    results = res.get("results", [])
                    if not results:
                        st.markdown(
                            '<div class="info-box">No results above threshold.</div>',
                            unsafe_allow_html=True,
                        )
                    for r in results:
                        st.markdown(
                            result_card(r["prompt_id"], r["similarity_score"],
                                        r["content_preview"]),
                            unsafe_allow_html=True,
                        )
                    lat = res.get("latency", {})
                    st.markdown(latency_html({
                        "Embed": f"{lat.get('embed_ms', '—')} ms",
                        "Search": f"{lat.get('search_ms', '—')} ms",
                        "Total": f"{lat.get('total_ms', '—')} ms",
                        "Hits": str(len(results)),
                    }), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Duplicate Clusters
# ══════════════════════════════════════════════════════════════════════════════
with tab4:

    # ── Controls row (narrow) ─────────────────────────────────────────────────
    col_ctrl, col_meta = st.columns([1, 2], gap="large")

    with col_ctrl:
        st.markdown('<div class="section-label">Clustering parameters</div>',
                    unsafe_allow_html=True)
        dup_thresh = st.slider(
            "Duplicate threshold",
            min_value=0.5, max_value=1.0, value=0.85, step=0.01,
            format="%.2f", key="dup_thresh",
        )
        st.markdown("""
        <div class="info-box">
            Complete-linkage agglomerative clustering. Two prompts are considered
            duplicates when their cosine similarity ≥ threshold. Higher values = stricter.
        </div>
        """, unsafe_allow_html=True)
        cluster_btn = st.button("⟢  Detect Clusters", key="dup_run")

    # ── Results — full-width below controls ───────────────────────────────────
    if cluster_btn:
        with st.spinner("Running clustering…"):
            res, err = api("get", "/api/analysis/duplicates",
                           params={"threshold": dup_thresh})
        if err:
            st.error(err)
        else:
            clusters = res.get("clusters", [])
            lat      = res.get("latency", {})

            if not clusters:
                st.markdown(
                    '<div class="info-box">No duplicate clusters found at this threshold.</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Latency + summary shown in the meta column
                with col_meta:
                    st.markdown('<div class="section-label">Run summary</div>',
                                unsafe_allow_html=True)
                    st.markdown(latency_html({
                        "Cluster time": f"{lat.get('cluster_ms', '—')} ms",
                        "Clusters": str(len(clusters)),
                        "Duplicate prompts": str(sum(len(c["prompts"]) for c in clusters)),
                        "Avg size": f"{sum(len(c['prompts']) for c in clusters)/len(clusters):.1f}",
                    }), unsafe_allow_html=True)

                # ── Full-width scatter ────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Dense space view — pan &amp; zoom to explore</div>',
                            unsafe_allow_html=True)
                fig = scatter_clusters(clusters)
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
                        "modeBarButtonsToAdd": ["pan2d"],
                        "scrollZoom": True,
                        "displaylogo": False,
                        "toImageButtonOptions": {
                            "format": "png", "filename": "clusters", "scale": 2,
                        },
                    },
                )

                # ── Cluster Cards ─────────────────────────────────────────────
                st.markdown('<div class="section-label">Cluster cards</div>',
                            unsafe_allow_html=True)
                # Render cards in 2 columns for breathing room
                card_cols = st.columns(2, gap="medium")
                for i, c in enumerate(clusters):
                    with card_cols[i % 2]:
                        render_cluster_card(c)