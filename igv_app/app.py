import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import streamlit as st

# ── Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Genomic Attribution Viewer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_NAMES  = [f"Class {i}" for i in range(5)]
CLASS_COLORS = ["#00BFFF", "#FF6B6B", "#7CFC00", "#FFD700", "#DA70D6"]
GRID_STEP    = 100_000
BG           = "#111111"
BG_DARK      = "#0d0d0d"

# ── Load precomputed data ───────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "igv_precomputed.npz")

@st.cache_resource
def load_data():
    raw = np.load(DATA_PATH, allow_pickle=False)
    chroms = sorted({k.split('_')[0] for k in raw.files if '_grid' in k},
                    key=lambda x: int(x) if x.isdigit() else 99)
    out = {}
    for c in chroms:
        chrom = f'chr{c}'
        out[chrom] = {
            'grid':      raw[f'{c}_grid'],
            'smoothed':  raw[f'{c}_smoothed'],
            'peaks_fdr': {cls: raw[f'{c}_fdr_{cls}'] for cls in range(5)},
            'peaks_idr': {cls: raw[f'{c}_idr_{cls}'] for cls in range(5)},
        }
    return out

chr_data = load_data()
chroms   = list(chr_data.keys())

# ── Custom dark CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #0d0d0d !important;
    color: #e0e0e0 !important;
}
[data-testid="stSidebar"] {
    background-color: #161616 !important;
    border-right: 1px solid #2a2a2a;
}
[data-testid="stSidebar"] * { color: #d0d0d0 !important; }
section.main > div { padding-top: 1rem; }
.stSlider > div > div > div { background: #333 !important; }
.stSelectbox label, .stSlider label, .stRadio label { color: #aaa !important; font-size: 0.82rem; }
h1 { color: #ffffff !important; font-size: 1.3rem !important; font-weight: 700 !important; }
h3 { color: #cccccc !important; font-size: 0.95rem !important; font-weight: 600 !important; margin-bottom: 0.2rem; }
.metric-box {
    background: #1a1a1a;
    border: 1px solid #2d2d2d;
    border-radius: 8px;
    padding: 10px 16px;
    text-align: center;
    margin-bottom: 6px;
}
.metric-box .val { font-size: 1.6rem; font-weight: 700; }
.metric-box .lbl { font-size: 0.72rem; color: #888; text-transform: uppercase; letter-spacing: 0.06em; }
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 IGV Browser")
    st.markdown("---")

    chrom = st.selectbox("Chromosome", chroms, index=0)

    data     = chr_data[chrom]
    grid     = data["grid"]
    smoothed = data["smoothed"]
    peaks_fdr = data["peaks_fdr"]
    peaks_idr = data["peaks_idr"]

    chrom_len_mb = (grid[-1] - grid[0]) / 1e6
    st.caption(f"Length: **{chrom_len_mb:.1f} Mb**")
    st.markdown("---")

    method = st.radio(
        "Peak method",
        ["FDR  (Benjamini–Hochberg)", "IDR  (Multi-Scale Rank)", "Both"],
        index=2,
    )

    st.markdown("---")
    st.markdown("### 🔬 Classes")
    class_toggles = {}
    for i, (name, col) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        class_toggles[i] = st.checkbox(
            name, value=True,
            key=f"cls_{i}",
            help=f"Toggle {name}",
        )

    st.markdown("---")
    st.markdown("### 📐 Window")
    win_mb = st.slider("Width (Mb)", 1.0, float(int(chrom_len_mb)), 10.0, 0.5)
    start_mb = st.slider(
        "Start (Mb)",
        float(grid[0] / 1e6),
        float(max(grid[0] / 1e6, grid[-1] / 1e6 - win_mb)),
        float(grid[0] / 1e6),
        0.1,
    )

    st.markdown("---")
    show_fdr_vs_idr = st.checkbox("Compare FDR vs IDR", value=False)

# ── Helpers ─────────────────────────────────────────────────────────────────
def pick_peaks(chrom_peaks_fdr, chrom_peaks_idr, method):
    if "FDR" in method and "IDR" not in method:
        return chrom_peaks_fdr, None
    if "IDR" in method and "FDR" not in method:
        return None, chrom_peaks_idr
    return chrom_peaks_fdr, chrom_peaks_idr

def draw_igv_track(ax, mb, smoothed, peaks_dict, active_classes,
                   method_label, method_color, show_label=True):
    active = [c for c in range(5) if active_classes.get(c, True)]
    if not active:
        ax.text(0.5, 0.5, "No classes selected", transform=ax.transAxes,
                color="#666", ha="center", va="center")
        return

    all_sigs = [smoothed[:, c] for c in active]
    g_ymin = min(s.min() for s in all_sigs)
    g_ymax = max(s.max() for s in all_sigs)
    span = g_ymax - g_ymin if g_ymax != g_ymin else 1e-6

    for cls in active:
        col = CLASS_COLORS[cls]
        sig = smoothed[:, cls]
        ax.plot(mb, sig, color=col, linewidth=0.9, alpha=0.9, zorder=3,
                label=CLASS_NAMES[cls])
        ax.fill_between(mb, 0, sig, where=(sig > 0), color=col, alpha=0.10, zorder=2)

        if peaks_dict is not None:
            for pidx in peaks_dict.get(cls, []):
                if pidx >= len(mb): continue
                rgn = ((mb >= mb[pidx] - GRID_STEP/1e6) &
                       (mb <= mb[pidx] + GRID_STEP/1e6))
                if rgn.any():
                    ax.fill_between(mb[rgn], 0, sig[rgn],
                                    color=col, alpha=0.60, zorder=4)

    ax.axhline(0, color="#444", linewidth=0.7, linestyle="--", zorder=1)
    ax.set_facecolor(BG)
    ax.set_xlim(mb[0], mb[-1])
    ax.set_ylim(g_ymin - span * 0.15, g_ymax + span * 0.22)
    ax.tick_params(colors="#888", labelsize=7)
    ax.set_ylabel("Attribution\n(median−1)", color="#aaa", fontsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a2a")

    if show_label and peaks_dict is not None:
        n_peaks = sum(len(peaks_dict.get(c, [])) for c in active)
        ax.text(0.005, 0.95,
                f"{method_label}  ·  {n_peaks} peaks (selected classes)",
                transform=ax.transAxes, color=method_color,
                fontsize=9, fontweight="bold", va="top",
                bbox=dict(facecolor="#1a1a1a", edgecolor=method_color,
                          boxstyle="round,pad=0.25", alpha=0.88))


# ── Main view ───────────────────────────────────────────────────────────────
view_start = start_mb
view_end   = start_mb + win_mb
mask = ((grid / 1e6) >= view_start) & ((grid / 1e6) <= view_end)
mb_w = grid[mask] / 1e6
sm_w = smoothed[mask, :]

# peak indices → mask into window
def window_peaks(peaks_dict, mask):
    local_idxs = np.where(mask)[0]
    result = {}
    for cls, pidxs in peaks_dict.items():
        win_pi = []
        for pidx in pidxs:
            where = np.where(local_idxs == pidx)[0]
            if len(where):
                win_pi.append(where[0])
        result[cls] = np.array(win_pi, dtype=int)
    return result

pf_w = window_peaks(peaks_fdr, mask)
pi_w = window_peaks(peaks_idr, mask)

# ── Header row ──────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([5, 1])
with hcol1:
    st.markdown(
        f"<h1>🧬 {chrom} &nbsp; "
        f"<span style='color:#888;font-size:0.95rem;font-weight:400'>"
        f"{view_start:.2f} – {view_end:.2f} Mb</span></h1>",
        unsafe_allow_html=True,
    )
with hcol2:
    total_fdr = sum(len(peaks_fdr.get(c, [])) for c in range(5))
    total_idr = sum(len(peaks_idr.get(c, [])) for c in range(5))
    st.markdown(
        f"<div class='metric-box'>"
        f"<div class='val' style='color:#FF6040'>{total_fdr}</div>"
        f"<div class='lbl'>FDR peaks</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='metric-box'>"
        f"<div class='val' style='color:#00E5FF'>{total_idr}</div>"
        f"<div class='lbl'>IDR peaks</div></div>",
        unsafe_allow_html=True,
    )

# ── Compare mode ────────────────────────────────────────────────────────────
if show_fdr_vs_idr:
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 6), facecolor=BG_DARK, sharex=True
    )
    fig.subplots_adjust(hspace=0.06, top=0.93, bottom=0.10, left=0.07, right=0.97)
    fig.suptitle(f"{chrom}  ·  FDR vs IDR  ·  {view_start:.1f}–{view_end:.1f} Mb",
                 color="white", fontsize=11, fontweight="bold")

    draw_igv_track(ax_top, mb_w, sm_w, pf_w, class_toggles,
                   "FDR (Benjamini–Hochberg)", "#FF6040")
    draw_igv_track(ax_bot, mb_w, sm_w, pi_w, class_toggles,
                   "IDR (Multi-Scale Rank)", "#00E5FF")

    ax_top.tick_params(labelbottom=False)
    ax_bot.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax_bot.set_xlabel("Genomic position (Mb)", color="#aaa", fontsize=9)

    # legend on top panel
    handles = [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
               for c in range(5) if class_toggles.get(c, True)]
    ax_top.legend(handles=handles, facecolor="#1a1a1a", edgecolor="#333",
                  labelcolor="white", fontsize=8, loc="upper right",
                  framealpha=0.9, ncol=5, handlelength=1.0)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

else:
    # ── Single track ──────────────────────────────────────────────────────
    pf_sel = pf_w if "FDR" in method else None
    pi_sel = pi_w if "IDR" in method else None

    fig, ax = plt.subplots(figsize=(14, 4), facecolor=BG_DARK)
    fig.subplots_adjust(top=0.90, bottom=0.15, left=0.07, right=0.97)

    active = [c for c in range(5) if class_toggles.get(c, True)]
    if active:
        all_sigs = [sm_w[:, c] for c in active]
        g_ymin = min(s.min() for s in all_sigs)
        g_ymax = max(s.max() for s in all_sigs)
        span = g_ymax - g_ymin if g_ymax != g_ymin else 1e-6

        for cls in active:
            col = CLASS_COLORS[cls]
            sig = sm_w[:, cls]
            ax.plot(mb_w, sig, color=col, linewidth=1.0, alpha=0.9,
                    zorder=3, label=CLASS_NAMES[cls])
            ax.fill_between(mb_w, 0, sig, where=(sig > 0),
                            color=col, alpha=0.10, zorder=2)

            for peaks_dict in [p for p in [pf_sel, pi_sel] if p is not None]:
                for pidx in peaks_dict.get(cls, []):
                    if pidx >= len(mb_w): continue
                    rgn = ((mb_w >= mb_w[pidx] - GRID_STEP/1e6) &
                           (mb_w <= mb_w[pidx] + GRID_STEP/1e6))
                    if rgn.any():
                        ax.fill_between(mb_w[rgn], 0, sig[rgn],
                                        color=col, alpha=0.60, zorder=4)

        ax.axhline(0, color="#444", linewidth=0.7, linestyle="--", zorder=1)
        ax.set_ylim(g_ymin - span*0.15, g_ymax + span*0.22)

    ax.set_facecolor(BG)
    ax.set_xlim(mb_w[0] if len(mb_w) else view_start,
                mb_w[-1] if len(mb_w) else view_end)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.set_xlabel("Genomic position (Mb)", color="#aaa", fontsize=9)
    ax.set_ylabel("Attribution (median−1)", color="#aaa", fontsize=9)
    ax.tick_params(colors="#888", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a2a")

    handles = [mpatches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
               for c in range(5) if class_toggles.get(c, True)]
    if handles:
        ax.legend(handles=handles, facecolor="#1a1a1a", edgecolor="#333",
                  labelcolor="white", fontsize=8.5, loc="upper right",
                  framealpha=0.9, ncol=5, handlelength=1.0)

    # method badge
    method_str = method.split("(")[0].strip()
    ax.text(0.005, 0.97, method_str, transform=ax.transAxes,
            color="#00E5FF" if "IDR" in method else "#FF6040",
            fontsize=9, fontweight="bold", va="top",
            bbox=dict(facecolor="#1a1a1a", edgecolor="#333",
                      boxstyle="round,pad=0.25", alpha=0.85))

    ax.set_title(
        f"{chrom}  ·  {view_start:.2f}–{view_end:.2f} Mb",
        color="white", fontsize=10, fontweight="bold", pad=6,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Peak table ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Peaks in this window")

show_fdr = "FDR" in method or show_fdr_vs_idr
show_idr = "IDR" in method or show_fdr_vs_idr

rows = []
for cls in range(5):
    if not class_toggles.get(cls, True):
        continue
    nf = len(pf_w.get(cls, []))
    ni = len(pi_w.get(cls, []))
    rows.append({
        "Class": CLASS_NAMES[cls],
        "FDR peaks (window)": nf if show_fdr else "—",
        "IDR peaks (window)": ni if show_idr else "—",
        "FDR total (chr)": len(peaks_fdr.get(cls, [])),
        "IDR total (chr)": len(peaks_idr.get(cls, [])),
    })

if rows:
    import pandas as pd
    df = pd.DataFrame(rows).set_index("Class")
    st.dataframe(df, use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='color:#444;font-size:0.72rem;text-align:center;margin-top:1rem'>"
    "GRCh38 · GENCODE v47 · Attribution tensor (150 samples × 10k SNPs × 5 classes)"
    "</div>",
    unsafe_allow_html=True,
)
