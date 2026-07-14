import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from stages.data import load_nz_data

log = logging.getLogger(__name__)

EU27_ISO3 = [
    'AUT', 'BEL', 'BGR', 'HRV', 'CYP', 'CZE', 'DNK', 'EST', 'FIN', 'FRA',
    'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD',
    'POL', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP', 'SWE',
]

def run_boxplots(winners: pd.DataFrame, bootstrap_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Generate boxplots of implied net-zero years for the top 20 emitters
    for each winner model.
    """
    if winners.empty or bootstrap_df.empty:
        log.warning("boxplots: winners or bootstrap_df is empty — skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("*.png"):
        f.unlink()
    
    nz_data = load_nz_data()
    announced_map = dict(zip(nz_data["iso3"], nz_data["nz_ghg_year"]))
    
    top20_emitters = ['CHN', 'USA', 'IND', 'RUS', 'BRA', 'IDN', 'JPN', 'IRN', 'MEX', 'SAU',
                      'DEU', 'CAN', 'KOR', 'TUR', 'AUS', 'PAK', 'ZAF', 'VNM', 'THA', 'GBR']
    countries_to_plot = sorted(top20_emitters + ['EUU'])
    
    log.info("Generating boxplots for top 20 valid emitters...")
    
    for idx, row in winners.iterrows():
        strategy = row["outer_strategy"]
        target = row["target"]
        branch = row["branch"]
        model_name = row["model_name"]
        
        tag_interpretability = f"{strategy}_{target}_{model_name}_{branch}"
        
        sub_all = bootstrap_df[
            (bootstrap_df["outer_strategy_source"] == strategy) &
            (bootstrap_df["target"] == target) &
            (bootstrap_df["branch"] == branch) &
            (bootstrap_df["model_name"] == model_name)
        ]

        sub = sub_all[sub_all["country"].isin(countries_to_plot)]

        pivot_df = sub.pivot(index="iteration", columns="country", values="tnz_implied")
        pivot_df = pivot_df.reindex(columns=countries_to_plot)

        if pivot_df.empty:
            log.warning(f"No prediction data found for model {tag_interpretability}")
            continue

        eu_avg = (
            sub_all[sub_all["country"].isin(EU27_ISO3)]
            .groupby("iteration")["tnz_implied"]
            .mean()
        )
        pivot_df["EUU"] = eu_avg.reindex(pivot_df.index)
            
        plt.close('all')
        plt.rcParams['figure.dpi'] = 300
        plt.figure(figsize=(14, 8))
        
        sns.boxplot(data=pivot_df, orient='h')
        
        plt.yticks(ticks=range(len(countries_to_plot)), labels=countries_to_plot)
        plt.xlabel('Predicted Net-Zero Year')
        plt.xlim(2020, 2140)
        plt.grid(True)
        
        for j, country_code in enumerate(countries_to_plot):
            actual_nz_year = announced_map.get(country_code)
            if actual_nz_year is not None and not pd.isna(actual_nz_year):
                plt.scatter(actual_nz_year, j, color='red', zorder=3)
            
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        
        plt.savefig(out_dir / f"{tag_interpretability}.png")
        plt.close(fig)
        
    log.info(f"All {len(winners)} boxplots successfully generated and saved in {out_dir}")
