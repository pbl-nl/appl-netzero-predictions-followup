"""
pca_components.py – Explains what the PCA country embedding components capture.

Triggered automatically from post-analysis when any final model has
country_pc* features in its selected feature set. Writes per-branch
Markdown reports with plots and tables.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

STRUCTURAL_VARIABLES = [
    "gdp_per_capita_constant_2015_us",
    "energy_intensity_mj_per_usd_ppp",
    "renewable_energy_share_tfc_pct",
    "renewable_electricity_share_pct",
    "coal_rents_of_gdp",
    "natural_gas_rents_of_gdp",
    "oil_rents_of_gdp",
    "control_of_corruption_estimate",
    "political_stability_and_absence_of_violence_terrorism_estimate",
    "regulatory_quality_estimate",
    "voice_and_accountability_estimate",
    "population_total",
    "population_density_people_per_sq_km_of_land_area",
    "total_hard_to_abate_emissions_per_capita",
    "total_ghg_emissions_per_capita",
    "non_co2_emissions_per_capita",
]

_COUNTRY_PC_PREFIX = "country_pc"
_SYNTHETIC_PREFIXES = ("country_pc", "year_std")
_PREPROCESSED_ROOT = Path("data/preprocessed")


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _discover_preprocessed_dir(branches: list[str]) -> Path | None:
    if not _PREPROCESSED_ROOT.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for d in _PREPROCESSED_ROOT.iterdir():
        if not d.is_dir():
            continue
        for branch in branches:
            if (d / "final" / branch / "preprocessor.joblib").exists():
                candidates.append((d.stat().st_mtime, d))
                break
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


def _strategy_col(final_results: pd.DataFrame) -> str:
    return "outer_strategy_source" if "outer_strategy_source" in final_results.columns else "outer_strategy"


def _load_selected_pca_features(
    preprocessed_dir: Path,
    final_results: pd.DataFrame,
) -> dict[tuple[str, str, str], list[str]]:
    """
    Returns {(strategy, branch, target): [country_pc* features selected by RFE]}.
    Combos with no country_pc* map to an empty list.
    """
    scol = _strategy_col(final_results)
    combos = final_results[[scol, "branch", "target"]].drop_duplicates()
    result: dict[tuple[str, str, str], list[str]] = {}
    for _, row in combos.iterrows():
        strategy, branch, target = str(row[scol]), str(row["branch"]), str(row["target"])
        path = (
            preprocessed_dir / "prepared" / "final"
            / strategy / branch / target / "selected_feature_columns.json"
        )
        if not path.exists():
            continue
        try:
            selected: list[str] = json.loads(path.read_text())
            result[(strategy, branch, target)] = [
                f for f in selected if f.startswith(_COUNTRY_PC_PREFIX)
            ]
        except Exception as exc:
            log.warning("pca_components: could not load %s: %s", path, exc)
    return result


# ---------------------------------------------------------------------------
# PCA re-derivation (loadings are not stored on the preprocessor)
# ---------------------------------------------------------------------------


def _rebuild_pca_loadings(
    preprocessor: Any,
    merged_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Re-fits PCA on the same country summary matrix the preprocessor used.
    Returns (explained_variance_ratio, components, summary_feature_names).
    components has shape (n_components, n_summary_features).
    """
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from stages.preprocessing import COUNTRY_COL, RANDOM_STATE, TEST_YEAR, YEAR_COL

    all_retained: list[str] = list(preprocessor.retained_continuous_features_)
    feats = [
        f for f in all_retained
        if not any(f.startswith(p) for p in _SYNTHETIC_PREFIXES)
        and f in merged_df.columns
    ]
    if not feats:
        return np.array([]), np.array([]), []

    train_df = merged_df[merged_df[YEAR_COL] < TEST_YEAR]
    grouped = train_df.groupby(COUNTRY_COL)[feats]
    medians = grouped.median()
    miss_rates = grouped.apply(lambda g: g.isna().mean())
    medians.columns = [f"med_{c}" for c in feats]
    miss_rates.columns = [f"miss_{c}" for c in feats]
    summary = pd.concat([medians, miss_rates], axis=1)
    summary_feature_names = list(summary.columns)

    n_countries = len(summary)
    n_components = min(
        int(preprocessor.country_n_pca_components_),
        max(n_countries - 1, 1),
        summary.shape[1],
    )
    if n_components < 1:
        return np.array([]), np.array([]), []

    summary_filled = SimpleImputer(strategy="median").fit_transform(summary.values)
    summary_scaled = StandardScaler().fit_transform(summary_filled)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(summary_scaled)
    return pca.explained_variance_ratio_, pca.components_, summary_feature_names


# ---------------------------------------------------------------------------
# Structural correlations
# ---------------------------------------------------------------------------


def _compute_structural_correlations(
    country_scores: pd.DataFrame,
    merged_df: pd.DataFrame,
) -> pd.DataFrame:
    from scipy.stats import spearmanr

    from stages.preprocessing import COUNTRY_COL, TEST_YEAR, YEAR_COL

    available = [c for c in STRUCTURAL_VARIABLES if c in merged_df.columns]
    if not available:
        return pd.DataFrame()

    country_struct = (
        merged_df[merged_df[YEAR_COL] < TEST_YEAR]
        .groupby(COUNTRY_COL)[available]
        .median(numeric_only=True)
    )
    joined = country_scores.join(country_struct, how="inner")

    rows: list[dict] = []
    for pc in country_scores.columns:
        for var in available:
            subset = joined[[pc, var]].dropna()
            if len(subset) < 5 or subset[pc].nunique() <= 1 or subset[var].nunique() <= 1:
                rho, p_val = float("nan"), float("nan")
            else:
                rho, p_val = spearmanr(subset[pc], subset[var])
                rho, p_val = float(rho), float(p_val)
            rows.append({"pc": pc, "variable": var, "spearman_r": rho, "p_value": p_val,
                         "n_countries": int(len(subset))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _plot_variance_explained(
    explained_ratios: np.ndarray,
    pc_names: list[str],
    out_dir: Path,
    branch: str,
) -> str:
    import matplotlib.pyplot as plt

    n = len(pc_names)
    ratios = explained_ratios[:n]
    colors = ["#2166ac" if i == 0 else "#92c5de" for i in range(n)]
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.5 * n)))
    bars = ax.barh(pc_names[::-1], (ratios * 100)[::-1], color=colors[::-1])
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xlabel("Variance Explained (%)")
    ax.set_title(f"PCA Variance Explained — {branch}")
    ax.set_xlim(0, float(ratios.max()) * 120 if ratios.size > 0 else 10)
    fig.tight_layout()
    fname = f"variance_explained_{branch}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def _plot_country_scores(
    country_scores: pd.DataFrame,
    pc: str,
    out_dir: Path,
    branch: str,
    n: int = 10,
) -> str:
    import matplotlib.pyplot as plt

    scores = country_scores[pc].dropna().sort_values()
    top_n = pd.concat([scores.head(n), scores.tail(n)]).drop_duplicates()
    colors = ["#d01c8b" if v < 0 else "#4dac26" for v in top_n.values]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(top_n))))
    ax.barh(top_n.index.astype(str), top_n.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"{pc} score")
    ax.set_title(f"Country Scores — {pc} ({branch})\nTop/bottom {n} countries")
    fig.tight_layout()
    fname = f"country_scores_{branch}_{pc}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def _plot_structural_heatmap(
    corr_df: pd.DataFrame,
    pc_names: list[str],
    out_dir: Path,
    branch: str,
) -> str:
    import matplotlib.pyplot as plt
    import seaborn as sns

    pivot = corr_df.pivot_table(index="pc", columns="variable", values="spearman_r", aggfunc="mean")
    pivot = pivot.reindex(index=pc_names)
    n_vars = len(pivot.columns)
    n_pcs = len(pivot)
    fig, ax = plt.subplots(figsize=(max(10, 0.8 * n_vars), max(3, 0.55 * n_pcs)))
    sns.heatmap(
        pivot, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax,
        annot=True, fmt=".2f", annot_kws={"size": 7}, linewidths=0.3,
    )
    ax.set_title(f"Spearman ρ: PCA Components × Structural Variables ({branch})")
    ax.set_xlabel("")
    ax.set_ylabel("Component")
    fig.tight_layout()
    fname = f"structural_correlations_{branch}.png"
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def _md_table(df: pd.DataFrame, float_fmt: str = ".4f", max_rows: int = 30) -> str:
    if df.empty:
        return "_No data._"
    sub = df.head(max_rows).copy()

    def _fmt(v: Any) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return format(float(v), float_fmt)
        return str(v)

    headers = list(sub.columns)
    rows = [[_fmt(cell) for cell in row] for row in sub.values]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    hdr = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    body = [
        "| " + " | ".join(r[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    ]
    return "\n".join([hdr, sep, *body])


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _write_branch_report(
    branch: str,
    pc_names_selected: list[str],
    all_pc_names: list[str],
    explained_ratios: np.ndarray,
    country_scores: pd.DataFrame,
    corr_df: pd.DataFrame,
    loadings: np.ndarray,
    loading_feature_names: list[str],
    model_table: pd.DataFrame,
    image_paths: dict[str, str],
    out_dir: Path,
) -> None:
    ev_df = pd.DataFrame({
        "component": all_pc_names,
        "variance_explained_pct": [
            f"{explained_ratios[i]*100:.2f}%" if i < len(explained_ratios) else "n/a"
            for i in range(len(all_pc_names))
        ],
    })

    sections: list[str] = [f"""# PCA Country Embedding Analysis — Branch: `{branch}`

## How to Interpret This Report

- **PC scores** represent where each country sits along a latent axis derived from its
  historical feature medians and data-missingness patterns. A high score on PC1 means
  the country is at the "high" end of whatever structural dimension that axis captures;
  a low score is the opposite.

- **Structural correlations** (Spearman ρ) measure how strongly each component aligns
  with a known indicator (GDP, emissions, governance…). |ρ| ≥ 0.5 is a strong relationship;
  |ρ| ≥ 0.3 is moderate. Positive ρ: higher-scoring countries are richer/more-developed on
  that variable. Negative ρ: the opposite.

- **Loadings** describe how each input feature (country median or missingness rate) contributes
  to the PC. A large positive loading means a high country median on that feature drives the
  PC score up. `med_*` features reflect levels; `miss_*` features reflect data-gap patterns.

- **Caveat on loadings**: Loadings are re-derived by re-fitting PCA on the same training data.
  They should match the training run numerically under the same random seed, but may drift
  slightly if the underlying raw data has changed.

---

## Variance Explained

{_md_table(ev_df)}

![variance_explained]({image_paths.get('variance_explained', '')})

---

## Component Analysis (selected components only)
"""]

    for i, pc in enumerate(all_pc_names):
        if pc not in pc_names_selected:
            continue

        ev_pct = float(explained_ratios[i]) * 100 if i < len(explained_ratios) else 0.0

        if not model_table.empty and "selected_pca_components" in model_table.columns:
            mask = model_table["selected_pca_components"].str.contains(pc, na=False)
            n_models = int(mask.sum())
            targets_using = sorted(model_table.loc[mask, "target"].unique().tolist())
        else:
            n_models, targets_using = 0, []

        scores_series = country_scores[pc].dropna().sort_values(ascending=False)
        top10 = scores_series.head(10).reset_index()
        top10.columns = ["country", "score"]
        bot10 = scores_series.tail(10).iloc[::-1].reset_index()
        bot10.columns = ["country", "score"]

        pc_corr = pd.DataFrame()
        if not corr_df.empty:
            pc_corr = corr_df[corr_df["pc"] == pc].copy()
            pc_corr["significant"] = pc_corr["p_value"].apply(
                lambda p: "yes" if (not pd.isna(p) and p < 0.05) else "no"
            )
            pc_corr = (
                pc_corr[["variable", "spearman_r", "p_value", "n_countries", "significant"]]
                .sort_values("spearman_r", ascending=False)
            )

        top_pos = pd.DataFrame()
        top_neg = pd.DataFrame()
        if len(loadings) > 0 and pc in all_pc_names:
            pc_idx = all_pc_names.index(pc)
            if pc_idx < len(loadings) and len(loading_feature_names) > 0:
                vec = loadings[pc_idx]
                load_df = pd.DataFrame({
                    "feature_type": [
                        "median" if f.startswith("med_") else "missingness"
                        for f in loading_feature_names
                    ],
                    "feature": [
                        f[4:] if f.startswith("med_") else f[5:]
                        for f in loading_feature_names
                    ],
                    "loading": vec,
                }).sort_values("loading", ascending=False)
                top_pos = load_df.head(10)
                top_neg = load_df.tail(10).iloc[::-1]

        img_key = f"country_scores_{pc}"
        sections.append(f"""
### `{pc}` — {ev_pct:.1f}% variance explained

**Selected by {n_models} model(s)**
Targets: {', '.join(targets_using) if targets_using else 'n/a'}

#### Top 10 Countries by Score (Highest)

{_md_table(top10)}

#### Top 10 Countries by Score (Lowest)

{_md_table(bot10)}

![country_scores_{pc}]({image_paths.get(img_key, '')})

#### Structural Variable Correlations (sorted by Spearman ρ, descending)

{_md_table(pc_corr)}

#### Top 10 Positive Feature Loadings

{_md_table(top_pos)}

#### Top 10 Negative Feature Loadings (most negative first)

{_md_table(top_neg)}

---
""")

    sections.append(f"""## Model Selection Summary

Which final models in this branch selected `country_pc*` features via RFE:

{_md_table(model_table, max_rows=60)}

![structural_correlations]({image_paths.get('structural_correlations', '')})
""")

    report_path = out_dir / f"pca_analysis_{branch}.md"
    report_path.write_text("\n".join(sections), encoding="utf-8")
    log.info("pca_components: written %s", report_path)


def _write_model_selection_summary(
    all_selection_rows: list[dict],
    out_dir: Path,
) -> None:
    if not all_selection_rows:
        return
    df = pd.DataFrame(all_selection_rows)
    report = f"""# PCA Component Selection Summary — All Models

This table shows which final models selected `country_pc*` features via RFE.
Models not listed here had no PCA components in their selected feature set.

{_md_table(df, max_rows=200)}
"""
    (out_dir / "model_pca_selection.md").write_text(report, encoding="utf-8")
    log.info("pca_components: written model_pca_selection.md")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_pca_components(
    final_results: pd.DataFrame,
    merged_df: pd.DataFrame | None,
    out_dir: Path,
    preprocessed_dir: Path | None = None,
) -> None:
    if final_results.empty:
        log.info("pca_components: final_results is empty — skipping.")
        return
    if merged_df is None:
        log.warning("pca_components: merged_df not available — skipping.")
        return

    # Step 1 — locate preprocessed cache
    if preprocessed_dir is None:
        branches = final_results["branch"].dropna().unique().tolist()
        preprocessed_dir = _discover_preprocessed_dir(branches)
    if preprocessed_dir is None:
        log.warning("pca_components: could not locate data/preprocessed/<hash>/ — skipping.")
        return
    preprocessed_dir = Path(preprocessed_dir)

    # Step 2 — collect selected PCA features per (strategy, branch, target)
    selection_map = _load_selected_pca_features(preprocessed_dir, final_results)
    if not selection_map:
        log.info("pca_components: no selected_feature_columns.json files found — skipping.")
        return

    # Step 3 — gate: does any model actually use country_pc* features?
    if not any(feats for feats in selection_map.values()):
        log.info("pca_components: no country_pc* features selected in any final model — skipping.")
        return

    log.info("pca_components: PCA country features detected; generating reports.")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build per-row model selection table (rows with PCA features only)
    scol = _strategy_col(final_results)
    all_selection_rows: list[dict] = []
    for _, row in final_results.iterrows():
        key = (str(row[scol]), str(row["branch"]), str(row["target"]))
        pca_feats = selection_map.get(key, [])
        if pca_feats:
            all_selection_rows.append({
                "branch": row["branch"],
                "target": row["target"],
                "strategy": row[scol],
                "model_name": row.get("model_name", ""),
                "candidate_rank": row.get("candidate_rank", ""),
                "selected_pca_components": ", ".join(sorted(pca_feats)),
            })

    _write_model_selection_summary(all_selection_rows, out_dir)

    # Step 4 — per-branch analysis
    for branch in sorted(final_results["branch"].dropna().unique()):
        branch_pca_feats: set[str] = set()
        for (s, b, t), feats in selection_map.items():
            if b == branch:
                branch_pca_feats.update(feats)
        if not branch_pca_feats:
            continue
        selected_pcs = sorted(branch_pca_feats)

        try:
            from stages.training.data_preparation import load_final_preprocessor
            preprocessor = load_final_preprocessor(preprocessed_dir, branch)
        except Exception as exc:
            log.warning("pca_components: failed to load preprocessor for branch=%s: %s", branch, exc)
            continue

        all_pc_names: list[str] = list(preprocessor.pca_component_names_)
        if not all_pc_names:
            log.info("pca_components: no pca_component_names_ on branch=%s preprocessor.", branch)
            continue

        embedding_map: dict[str, np.ndarray] = preprocessor.country_embedding_map_
        if not embedding_map:
            log.info("pca_components: empty country_embedding_map_ for branch=%s.", branch)
            continue
        country_scores = pd.DataFrame.from_dict(
            embedding_map, orient="index", columns=all_pc_names
        )
        country_scores.index.name = "country"

        explained_ratios: np.ndarray = np.array([])
        loadings: np.ndarray = np.array([])
        loading_feature_names: list[str] = []
        try:
            explained_ratios, loadings, loading_feature_names = _rebuild_pca_loadings(
                preprocessor, merged_df
            )
        except Exception as exc:
            log.warning("pca_components: loading re-derivation failed for branch=%s: %s", branch, exc)

        corr_df = pd.DataFrame()
        try:
            corr_df = _compute_structural_correlations(country_scores[selected_pcs], merged_df)
        except Exception as exc:
            log.warning("pca_components: correlation computation failed for branch=%s: %s", branch, exc)

        branch_rows = [r for r in all_selection_rows if r["branch"] == branch]
        model_table = pd.DataFrame(branch_rows) if branch_rows else pd.DataFrame()

        image_paths: dict[str, str] = {}

        if len(explained_ratios) > 0:
            try:
                image_paths["variance_explained"] = _plot_variance_explained(
                    explained_ratios, all_pc_names, out_dir, branch
                )
            except Exception as exc:
                log.warning("pca_components: variance plot failed for branch=%s: %s", branch, exc)

        for pc in selected_pcs:
            try:
                image_paths[f"country_scores_{pc}"] = _plot_country_scores(
                    country_scores, pc, out_dir, branch
                )
            except Exception as exc:
                log.warning("pca_components: country score plot failed %s/%s: %s", branch, pc, exc)

        if not corr_df.empty:
            try:
                image_paths["structural_correlations"] = _plot_structural_heatmap(
                    corr_df, selected_pcs, out_dir, branch
                )
            except Exception as exc:
                log.warning("pca_components: heatmap failed for branch=%s: %s", branch, exc)

        try:
            _write_branch_report(
                branch=branch,
                pc_names_selected=selected_pcs,
                all_pc_names=all_pc_names,
                explained_ratios=explained_ratios,
                country_scores=country_scores,
                corr_df=corr_df,
                loadings=loadings,
                loading_feature_names=loading_feature_names,
                model_table=model_table,
                image_paths=image_paths,
                out_dir=out_dir,
            )
        except Exception as exc:
            log.warning("pca_components: report writing failed for branch=%s: %s", branch, exc)
