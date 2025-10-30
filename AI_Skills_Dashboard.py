# =========================
# Imports
# =========================
import os
from datetime import datetime
import math
import re

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------- Page setup ----------
st.set_page_config(page_title="Analysis of AI Skills", layout="wide")

# =========================
# UMaine-inspired color palette
# =========================
PALETTE = {
    "maine_navy":   "#0F3D73",
    "maine_blue":   "#1E65A7",
    "maine_ltblue": "#79BDE8",
    "pine_dark":    "#1E7456",
    "pine_med":     "#2FA07B",
    "pine_light":   "#94D1BE",
    "paper":        "#FFFFFF",
    "paper_alt":    "#F5F7FA",
    "ink":          "#0B2239"
}

def cmap_from_hexes(name, hexes):
    return LinearSegmentedColormap.from_list(name, hexes)

C_BLUE  = cmap_from_hexes("UMaineBlues",  [PALETTE["maine_navy"], PALETTE["maine_blue"], PALETTE["maine_ltblue"]])
C_PINE  = cmap_from_hexes("UMainePines",  [PALETTE["pine_dark"],  PALETTE["pine_med"],  PALETTE["pine_light"]])

BORDER_HEX = PALETTE["maine_navy"]
ARROW_HEX  = PALETTE["maine_navy"]

# =========================
# Data (single Excel with skills + postings)
# =========================
from pathlib import Path
import streamlit as st
import pandas as pd

APP_DIR = Path(__file__).parent
data_path = APP_DIR / "AI_Skill_Dashboard_Data.xlsx"

if not data_path.exists():
    st.error(f"Data file not found at: {data_path}\nFiles in app dir: {[p.name for p in APP_DIR.iterdir()]}")
    st.stop()

df = pd.read_excel(data_path, engine="openpyxl")


def _norm_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

def _split_skills_any_delim(val) -> list[str]:
    tokens = re.split(r"[;|,]", str(val))
    return [t.strip().lower() for t in tokens if t.strip()]

@st.cache_data
def load_excel_data(xlsx_path: str):
    """
    Detect a skills sheet (Skill, L1_Cat, L2_Cat) and a postings sheet.
    Postings sheet may use 'Company Name' instead of 'Company' and may list one 'Skill' per row.
    We roll postings up so each job has 'Skills' as a semicolon-separated list.
    """
    xls = pd.ExcelFile(xlsx_path)

    skills_df = None
    postings_raw = None

    # Detect candidate sheets by columns (Sheet names can be anything)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        cols = {c.strip() for c in df.columns.astype(str)}

        # Skills sheet
        if skills_df is None and {"Skill", "L1_Cat", "L2_Cat"}.issubset(cols):
            skills_df = df.copy()

        # Postings sheet: accept either 'Company' or 'Company Name'
        postings_variants = [
            {"JobTitle", "Company", "Location", "DatePosted", "URL"},
            {"JobTitle", "Company Name", "Location", "DatePosted", "URL"},
        ]
        if postings_raw is None and any(variant.issubset(cols) for variant in postings_variants):
            postings_raw = df.copy()

    if skills_df is None:
        # Try first sheet as fallback if it has the right columns
        tmp = pd.read_excel(xls, sheet_name=0)
        if {"Skill", "L1_Cat", "L2_Cat"}.issubset({c.strip() for c in tmp.columns.astype(str)}):
            skills_df = tmp.copy()

    if skills_df is None:
        raise ValueError("Could not find a sheet with columns: Skill, L1_Cat, L2_Cat.")

    # ---- Clean SKILLS ----
    skills_df = skills_df.rename(columns=lambda c: c.strip())
    for col in ["Skill", "L1_Cat", "L2_Cat"]:
        if col not in skills_df.columns:
            raise ValueError(f"Skills sheet missing required column: {col}")
    skills_df["L1_Cat"] = skills_df["L1_Cat"].fillna("Unclassified").astype(str).str.strip()
    skills_df["L2_Cat"] = skills_df["L2_Cat"].fillna("Unclassified").astype(str).str.strip()
    skills_df["Skill"]  = skills_df["Skill"].astype(str).str.strip()

    # ---- Clean & roll-up POSTINGS (if present) ----
    postings_df = None
    if postings_raw is not None:
        pr = postings_raw.rename(columns=lambda c: str(c).strip())

        # Convert all text-like columns to strings safely
        for col in pr.columns:
            if pr[col].dtype != 'O':  # not already object/string
                pr[col] = pr[col].astype(str)
            pr[col] = pr[col].fillna("").astype(str).str.strip()

        # Normalize Company column
        if "Company" not in pr.columns and "Company Name" in pr.columns:
            pr = pr.rename(columns={"Company Name": "Company"})

        # Normalize categories if included
        for cat_col in ["L1_Cat", "L2_Cat"]:
            if cat_col in pr.columns:
                pr[cat_col] = _norm_series(pr[cat_col]).replace({"nan": np.nan})

        # Ensure minimal expected columns exist
        for col in ["JobTitle", "Company", "Location", "DatePosted", "URL"]:
            if col not in pr.columns:
                pr[col] = np.nan

        # Parse dates
        def _to_date(x):
            try:
                d = pd.to_datetime(x, errors="coerce")
                return d.dt.date if hasattr(d, "dt") else (d.date() if pd.notna(d) else pd.NaT)
            except Exception:
                return pd.NaT
        pr["DatePosted"] = pr["DatePosted"].apply(_to_date)

        # If postings have one Skill per row (as in your Sheet2), roll them up
        # Define the key that identifies a unique job posting
        key_cols = ["JobTitle", "Company", "Location", "DatePosted", "URL"]
        existing_cols = [c for c in key_cols if c in pr.columns]

        # Aggregate skills -> 'Skills'; keep first non-null L1_Cat/L2_Cat if present
        def _agg_skills(group):
            # collect Skill column if present; else use Skills if present already
            skills_single = group["Skill"].dropna().astype(str).str.strip() if "Skill" in group.columns else pd.Series([], dtype=str)
            skills_multi  = group["Skills"].dropna().astype(str).str.strip() if "Skills" in group.columns else pd.Series([], dtype=str)
            all_skills = list(skills_single) + list(skills_multi)
            if not all_skills:
                skills_out = ""
            else:
                # unique, preserve order
                seen = set()
                uniq = [s for s in all_skills if not (s in seen or seen.add(s))]
                skills_out = "; ".join(uniq)

            out = {
                "Skills": skills_out
            }
            # carry forward categories if present
            if "L1_Cat" in group.columns:
                out["L1_Cat"] = group["L1_Cat"].dropna().iloc[0] if group["L1_Cat"].notna().any() else np.nan
            if "L2_Cat" in group.columns:
                out["L2_Cat"] = group["L2_Cat"].dropna().iloc[0] if group["L2_Cat"].notna().any() else np.nan
            return pd.Series(out)

        postings_df = (
            pr.groupby(existing_cols, dropna=False, as_index=False)
              .apply(_agg_skills)
              .reset_index(drop=True)
        )

        # Tokenize Skills for overlap fallback
        postings_df["_skills_list"] = postings_df["Skills"].fillna("").apply(_split_skills_any_delim)

    return skills_df, postings_df

df, df_postings = load_excel_data(data_path)

# =========================
# Styling (cards, arrow, etc.)
# =========================
st.markdown(
    f"""
    <style>
        :root {{
            --card-h: 380px;
            --maine-navy: {BORDER_HEX};
            --paper: {PALETTE["paper"]};
            --text-ink: {PALETTE["ink"]};
        }}

        [data-testid="stHorizontalBlock"] {{ align-items: stretch !important; }}
        [data-testid="column"] {{ display: flex; }}
        [data-testid="column"] > div {{ flex: 1 1 auto; display: flex; }}

        .equal-col {{ display: flex; width: 100%; }}

        .card {{
            background: var(--paper);
            border: 2px solid var(--maine-navy);
            border-radius: 16px;
            padding: 18px 18px 8px 18px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.08);
            display: flex;
            flex-direction: column;
            height: var(--card-h);
            width: 100%;
        }}

        .card h3 {{
            margin: 0 0 12px 0;
            font-size: 1.1rem;
            color: var(--maine-navy);
            letter-spacing: 0.2px;
        }}

        .list {{
            margin-top: 6px;
            line-height: 1.6;
            font-size: 0.98rem;
            overflow: auto;
        }}

        .list a {{ color: var(--maine-navy); text-decoration: none; }}
        .list a:hover {{ text-decoration: underline; }}

        .centered-subheading {{
            text-align: center;
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 0.25rem;
            margin-bottom: 1rem;
            color: var(--text-ink);
        }}

        .arrow-wrap {{
            display: flex; align-items: center; justify-content: center;
            height: var(--card-h); width: 100%;
        }}

        .arrow-wrap svg {{ width: 140px; max-width: 18vw; height: auto; }}

        .spacer {{ height: 16px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Helpers (colors + pies + routing)
# =========================
def cmap_n(cmap, n, start=0.15, end=0.95):
    return [cmap(x) for x in np.linspace(start, end, max(n, 1))]

def prepare_counts(series, min_pct_for_other=0):
    counts = series.value_counts(dropna=False).copy()
    if min_pct_for_other > 0:
        total = counts.sum()
        small = counts[counts / total * 100 < min_pct_for_other]
        if len(small) > 0:
            counts = counts.drop(small.index)
            counts.loc["Other"] = small.sum()
    return counts.sort_values(ascending=False)

def pie_with_outside_labels(ax, counts, title=None, cmap=C_BLUE, min_pct_label=1.0):
    """Robust pie with outside labels; safe for single-slice and weird angles."""
    values = counts.values
    labels = counts.index.tolist()
    total = values.sum()

    if total == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return

    cols = cmap_n(cmap, len(values))
    wedges, _ = ax.pie(
        values,
        startangle=90,
        counterclock=False,
        colors=cols,
        radius=1.0,
        wedgeprops=dict(linewidth=1, edgecolor="white"),
    )

    if len(wedges) == 1:
        label_text = f"{labels[0]} — 100.0%"
        ax.text(0, 0, label_text, ha="center", va="center", fontsize=11)
        if title:
            ax.set_title(title, pad=10)
        ax.axis("equal")
        return

    for i, w in enumerate(wedges):
        pct = (values[i] / total) * 100
        if pct < min_pct_label:
            continue

        ang = (w.theta2 + w.theta1) / 2.0
        ang_rad = np.deg2rad(ang)
        x, y = np.cos(ang_rad), np.sin(ang_rad)

        r_label = 1.18
        xa, ya = r_label * np.sign(x), r_label * y
        ha = "left" if x > 0 else "right"
        label_text = f"{labels[i]} — {pct:.1f}%"

        arrow_kw = dict(
            arrowstyle="-",
            linewidth=0.8,
            color="#666",
            shrinkA=0, shrinkB=0,
            connectionstyle="arc3,rad=0.2",
        )

        try:
            ax.annotate(
                label_text,
                xy=(x, y),
                xytext=(xa, ya),
                ha=ha, va="center",
                fontsize=10,
                arrowprops=arrow_kw,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
            )
        except Exception:
            ax.text(xa * 1.05, ya * 1.05, label_text, ha=ha, va="center", fontsize=10)

    if title:
        ax.set_title(title, pad=10)
    ax.axis("equal")

def get_view():
    try:
        qp = st.query_params
        if isinstance(qp.get("view", None), list):
            return qp.get("view", [None])[0]
        return qp.get("view", None)
    except Exception:
        qp = st.experimental_get_query_params()
        return (qp.get("view", [None]) or [None])[0]

def back_home_button():
    if st.button("🏠 Back to Home"):
        try:
            st.query_params.clear()
        except Exception:
            st.experimental_set_query_params()
        st.rerun()

# =========================
# Postings helpers (from Excel)
# =========================
def filter_postings_for_L1(postings: pd.DataFrame, l1_value: str, skills_subset: pd.Series, limit: int = 12):
    """Prefer direct L1_Cat filter if present; fall back to skills overlap."""
    if postings is None or postings.empty:
        return pd.DataFrame()

    if "L1_Cat" in postings.columns and postings["L1_Cat"].notna().any():
        out = postings[postings["L1_Cat"] == l1_value].copy()
    else:
        wanted = set(s.lower().strip() for s in skills_subset.dropna().unique())
        if not wanted:
            return pd.DataFrame()
        overlaps = []
        for idx, row in postings.iterrows():
            rowset = set(row.get("_skills_list", []))
            k = len(wanted.intersection(rowset))
            if k > 0:
                overlaps.append((idx, k))
        if not overlaps:
            return pd.DataFrame()
        overlaps.sort(key=lambda x: x[1], reverse=True)
        idxs = [i for i, _ in overlaps[:limit]]
        out = postings.loc[idxs].copy()
        out["MatchedSkillsCount"] = [s for _, s in overlaps[:limit]]

    if "MatchedSkillsCount" in out.columns and "DatePosted" in out.columns:
        out = out.sort_values(["MatchedSkillsCount", "DatePosted"], ascending=[False, False])
    elif "MatchedSkillsCount" in out.columns:
        out = out.sort_values(["MatchedSkillsCount"], ascending=[False])
    elif "DatePosted" in out.columns:
        out = out.sort_values(["DatePosted"], ascending=False)

    return out.head(limit)

def filter_postings_for_L2(postings: pd.DataFrame, l2_value: str, skills_subset: pd.Series, limit: int = 12):
    """Prefer direct L2_Cat filter if present; fall back to skills overlap."""
    if postings is None or postings.empty:
        return pd.DataFrame()

    if "L2_Cat" in postings.columns and postings["L2_Cat"].notna().any():
        out = postings[postings["L2_Cat"] == l2_value].copy()
    else:
        wanted = set(s.lower().strip() for s in skills_subset.dropna().unique())
        if not wanted:
            return pd.DataFrame()
        overlaps = []
        for idx, row in postings.iterrows():
            rowset = set(row.get("_skills_list", []))
            k = len(wanted.intersection(rowset))
            if k > 0:
                overlaps.append((idx, k))
        if not overlaps:
            return pd.DataFrame()
        overlaps.sort(key=lambda x: x[1], reverse=True)
        idxs = [i for i, _ in overlaps[:limit]]
        out = postings.loc[idxs].copy()
        out["MatchedSkillsCount"] = [s for _, s in overlaps[:limit]]

    if "MatchedSkillsCount" in out.columns and "DatePosted" in out.columns:
        out = out.sort_values(["MatchedSkillsCount", "DatePosted"], ascending=[False, False])
    elif "MatchedSkillsCount" in out.columns:
        out = out.sort_values(["MatchedSkillsCount"], ascending=[False])
    elif "DatePosted" in out.columns:
        out = out.sort_values(["DatePosted"], ascending=False)

    return out.head(limit)

def render_postings_grid(dfp: pd.DataFrame, header: str, per_row: int = 2):
    st.subheader(header)
    if dfp is None or dfp.empty:
        st.info("No matching job postings found for this set of skills.")
        return

    rows = dfp.to_dict("records")
    n = len(rows)
    for i in range(0, n, per_row):
        cols = st.columns(per_row, gap="large")
        chunk = rows[i : i + per_row]
        for j, rec in enumerate(chunk):
            with cols[j]:
                st.markdown("----")
                title = (rec.get("JobTitle") or "Job posting").strip()
                st.markdown(f"**{title}**")
                parts = []
                company = (rec.get("Company") or "").strip()
                location = (rec.get("Location") or "").strip()
                datep = rec.get("DatePosted", None)
                matched = rec.get("MatchedSkillsCount", None)
                if company: parts.append(company)
                if location: parts.append(location)
                if pd.notna(datep) and datep != "":
                    parts.append(str(datep))
                if matched:
                    parts.append(f"{int(matched)} matched skill(s)")
                if parts:
                    st.caption(" • ".join(parts))
                url = (rec.get("URL") or "").strip()
                if url:
                    try:
                        st.link_button("Open posting", url)
                    except Exception:
                        st.markdown(f"[Open posting]({url})")

# =========================
# Dashboards (same layout as before) + postings grid
# =========================
def render_l1_skillonly_dashboard(df_all, l1_value: str, title_prefix: str, cmap):
    back_home_button()
    st.title(f"{title_prefix} — Summary")

    dfx = df_all[df_all["L1_Cat"] == l1_value].copy()
    st.write(f"**Total {title_prefix}:** {len(dfx):,}")
    if dfx.empty:
        st.info(f'No skills found for "{title_prefix}". Check L1_Cat values.')
        return

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader(f"{title_prefix} (comprehensive list)")
        skills_only = dfx[["Skill"]].sort_values("Skill")
        st.dataframe(skills_only, use_container_width=True, hide_index=True)
        st.download_button(
            label=f"⬇️ Download {title_prefix} (CSV)",
            data=skills_only.to_csv(index=False).encode("utf-8"),
            file_name=f"{l1_value.replace(' ', '_')}_skills.csv",
            mime="text/csv",
        )

    with right:
        st.subheader(f"Breakdown of {title_prefix}")
        counts = prepare_counts(dfx["L2_Cat"], min_pct_for_other=0)
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        pie_with_outside_labels(ax, counts, title=None, cmap=cmap, min_pct_label=1.0)
        st.pyplot(fig)

    posts = filter_postings_for_L1(df_postings, l1_value, dfx["Skill"], limit=12)
    render_postings_grid(posts, header="Example job postings demanding these skills", per_row=2)

def render_l2_skillonly_dashboard(df_all, l2_value: str, title_prefix: str, cmap):
    back_home_button()
    st.title(f"{title_prefix} — Summary")

    dfx = df_all[df_all["L2_Cat"] == l2_value].copy()
    st.write(f"**Total {title_prefix}:** {len(dfx):,}")
    if dfx.empty:
        st.info(f'No skills found for "{title_prefix}". Check L2_Cat values.')
        return

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader(f"{title_prefix}")
        skills_only = dfx[["Skill"]].sort_values("Skill")
        st.dataframe(skills_only, use_container_width=True, hide_index=True)
        st.download_button(
            label=f"⬇️ Download {title_prefix} (CSV)",
            data=skills_only.to_csv(index=False).encode("utf-8"),
            file_name=f"{l2_value.replace(' ', '_')}_skills.csv",
            mime="text/csv",
        )

    with right:
        st.subheader(f"Breakdown of {title_prefix}")
        counts = prepare_counts(dfx["L1_Cat"], min_pct_for_other=0)
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        pie_with_outside_labels(ax, counts, title=None, cmap=cmap, min_pct_label=1.0)
        st.pyplot(fig)

    posts = filter_postings_for_L2(df_postings, l2_value, dfx["Skill"], limit=12)
    render_postings_grid(posts, header="Example job postings demanding these skills", per_row=2)

# =========================
# Router
# =========================
PAGES = {
    # L1 category dashboards (UMaine blues)
    "ai_skills":       lambda d: render_l1_skillonly_dashboard(d, "AI Skills", "AI Skills", C_BLUE),
    "use_ai":          lambda d: render_l1_skillonly_dashboard(d, "Skills that Use AI", "Skills that Use AI", C_BLUE),
    "related_ai":      lambda d: render_l1_skillonly_dashboard(d, "Skills Related to AI", "Skills Related to AI", C_BLUE),
    "high_tech":       lambda d: render_l1_skillonly_dashboard(d, "High-Tech Skills", "High-Tech Skills", C_BLUE),
    "foundational":    lambda d: render_l1_skillonly_dashboard(d, "Foundational Skills", "Foundational Skills", C_BLUE),
    "domain_specific": lambda d: render_l1_skillonly_dashboard(d, "Domain-Specific Skills", "Domain-Specific Skills", C_BLUE),

    # L2 subcategory dashboards (UMaine pine greens)
    "skill_competency": lambda d: render_l2_skillonly_dashboard(d, "Skill/Competency", "Skill/Competency", C_PINE),
    "knowledge_domain": lambda d: render_l2_skillonly_dashboard(d, "Knowledge Domain", "Knowledge Domain", C_PINE),
    "certification":    lambda d: render_l2_skillonly_dashboard(d, "Certification/License", "Certification/License", C_PINE),
    "software_tool":    lambda d: render_l2_skillonly_dashboard(d, "Software Tool", "Software Tool", C_PINE),
    "occupation_role":  lambda d: render_l2_skillonly_dashboard(d, "Occupation/Role", "Occupation/Role", C_PINE),
}

view = get_view()
if view in PAGES:
    PAGES[view](df)
    st.stop()

# =========================
# HOME PAGE (cards + arrow + pies)
# =========================
st.title("Analysis of AI Skills")
st.markdown('<div class="centered-subheading">AI Skill Classification System</div>', unsafe_allow_html=True)

# First row
left, mid, right = st.columns([1, 0.2, 1], gap="large")

with left:
    st.markdown(
        f"""
        <div class="equal-col">
          <div class="card">
            <h3>Skill Category</h3>
            <ul class="list">
                <li><a href="?view=ai_skills" target="_self" rel="noopener">AI Skills – Core competencies required to design, develop, and build artificial intelligence systems.</a></li>
                <li><a href="?view=use_ai" target="_self" rel="noopener">Skills that Use AI – Abilities that involve leveraging existing AI tools, outputs, or systems in practice.</a></li>
                <li><a href="?view=related_ai" target="_self" rel="noopener">Skills Related to AI – Supporting capabilities that make AI possible, such as data infrastructure, mathematics, or computing power.</a></li>
                <li><a href="?view=high_tech" target="_self" rel="noopener">High-Tech Skills – Emerging and advanced technical skills not directly part of AI, but still at the leading edge of technology.</a></li>
                <li><a href="?view=foundational" target="_self" rel="noopener">Foundational Skills – Broad competencies that apply across domains.</a></li>
                <li><a href="?view=domain_specific" target="_self" rel="noopener">Domain-Specific Skills – Knowledge and skills tailored to particular industries or fields.</a></li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with mid:
    st.markdown(
        f"""
        <div class="equal-col">
          <div class="arrow-wrap">
            <svg viewBox="0 0 400 80" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Flow arrow">
              <rect x="10" y="30" width="300" height="20" rx="10" fill="{ARROW_HEX}"></rect>
              <polygon points="310,10 390,40 310,70" fill="{ARROW_HEX}"></polygon>
            </svg>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
        <div class="equal-col">
          <div class="card">
            <h3>Skill Subcategory</h3>
            <ul class="list">
                <li><a href="?view=skill_competency" target="_self" rel="noopener">Skill/Competency – A specific ability or learned behavior used to perform a task effectively.</a></li>
                <li><a href="?view=knowledge_domain" target="_self" rel="noopener">Knowledge Domain – A field of subject-matter expertise or conceptual understanding.</a></li>
                <li><a href="?view=certification" target="_self" rel="noopener">Certification/License – A formal credential verifying professional or regulatory qualification.</a></li>
                <li><a href="?view=software_tool" target="_self" rel="noopener">Software Tool – A digital program or platform used to complete or support a task.</a></li>
                <li><a href="?view=occupation_role" target="_self" rel="noopener">Occupation/Role – A defined job function combining related skills and responsibilities.</a></li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Second row (pies on home)
st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
left2, mid2, right2 = st.columns([1, 0.2, 1], gap="large")

with left2:
    st.subheader("Breakdown of AI Skills by Category")
    l1_counts_home = prepare_counts(df["L1_Cat"], min_pct_for_other=0)
    fig1, ax1 = plt.subplots(figsize=(6.2, 5.2))
    pie_with_outside_labels(ax1, l1_counts_home, title=None, cmap=C_BLUE, min_pct_label=1.0)
    st.pyplot(fig1)

with right2:
    st.subheader("Breakdown of AI Skills by Subcategory")
    l2_counts_home = prepare_counts(df["L2_Cat"], min_pct_for_other=0)
    fig2, ax2 = plt.subplots(figsize=(6.2, 5.2))
    pie_with_outside_labels(ax2, l2_counts_home, title=None, cmap=C_PINE, min_pct_label=1.0)
    st.pyplot(fig2)