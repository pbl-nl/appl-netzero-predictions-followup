"""Configuration dataclass for data cleaning."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CleaningConfig:
    """Configuration for cleaning merged indicator data."""

    step_name: str = "cleaning"
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data"
    )
    input_filename: str = "merged_indicators.csv"
    cleaned_filename: str = "cleaned_indicators.csv"
    diagnostics_filename: str = "cleaning_diagnostics.csv"
    report_dirname: str = "cleaning_report"
    report_markdown_filename: str = "cleaning_report.md"
    report_overview_filename: str = "overview.csv"
    report_by_issue_type_filename: str = "issues_by_type.csv"
    report_by_severity_filename: str = "issues_by_severity.csv"
    report_by_column_filename: str = "issues_by_column.csv"
    report_by_country_filename: str = "issues_by_country.csv"
    report_by_year_filename: str = "issues_by_year.csv"
    report_top_rows_filename: str = "top_problem_rows.csv"
    report_missingness_by_column_filename: str = "missingness_by_column.csv"
    top_n_rankings: int = 25

    panel_year_start: int = 1960
    panel_year_end: int = 2024
    drop_outside_panel_years: bool = True

    outlier_mad_threshold: float = 3.5
    outlier_iqr_multiplier: float = 3.0
    temporal_jump_threshold: float = 1.0

    non_negative_columns: tuple[str, ...] = (
        "total_ghg_emissions",
        "non_co2_emissions",
        "total_hard_to_abate_emissions",
        "population_total",
        "population_density_people_per_sq_km_of_land_area",
        "inform_risk_index",
        "rsf_press_freedom_index",
        "gdp_per_capita_constant_2015_us",
        "energy_intensity_mj_per_usd_ppp",
        "average_supply_of_protein_of_animal_origin_g_cap_day_3_year_average",
    )

    bounded_columns: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "agricultural_land_of_land_area": (0, 100),
        "forest_area_of_land_area": (0, 100),
        "coal_rents_of_gdp": (0, 100),
        "natural_gas_rents_of_gdp": (0, 100),
        "oil_rents_of_gdp": (0, 100),
        "renewable_energy_share_tfc_pct": (0, 100),
        "renewable_electricity_share_pct": (0, 100),
        "rsf_press_freedom_index": (0, 100),
        "inform_risk_index": (0, 10),
        "v2x_regime": (0, 3),
    })

    exclude_from_outlier_detection: tuple[str, ...] = (
        "v2x_regime",
    )

    temporal_jump_thresholds: dict[str, float] = field(default_factory=lambda: {
        "population_total": 0.10,
        "population_density_people_per_sq_km_of_land_area": 0.10,
        "agricultural_land_of_land_area": 0.20,
        "forest_area_of_land_area": 0.20,
        "ghg_growth": 5.0,
        "annual_deforestation_of_change": 5.0,
        "coal_rents_of_gdp": 3.0,
        "natural_gas_rents_of_gdp": 3.0,
        "oil_rents_of_gdp": 3.0,
    })
    temporal_jump_default_threshold: float = 1.0

    cross_column_rules: tuple[tuple[str, str], ...] = (
        ("non_co2_emissions", "total_ghg_emissions"),
        ("total_hard_to_abate_emissions", "total_ghg_emissions"),
    )

    constant_series_min_length: int = 10

    @property
    def input_path(self) -> Path:
        return self.data_dir / self.input_filename

    @property
    def cleaned_path(self) -> Path:
        return self.data_dir / self.cleaned_filename

    @property
    def diagnostics_path(self) -> Path:
        return self.data_dir / self.diagnostics_filename

    @property
    def report_dir(self) -> Path:
        return self.data_dir / self.report_dirname

    @property
    def report_markdown_path(self) -> Path:
        return self.report_dir / self.report_markdown_filename

    @property
    def report_overview_path(self) -> Path:
        return self.report_dir / self.report_overview_filename

    @property
    def report_by_issue_type_path(self) -> Path:
        return self.report_dir / self.report_by_issue_type_filename

    @property
    def report_by_severity_path(self) -> Path:
        return self.report_dir / self.report_by_severity_filename

    @property
    def report_by_column_path(self) -> Path:
        return self.report_dir / self.report_by_column_filename

    @property
    def report_by_country_path(self) -> Path:
        return self.report_dir / self.report_by_country_filename

    @property
    def report_by_year_path(self) -> Path:
        return self.report_dir / self.report_by_year_filename

    @property
    def report_top_rows_path(self) -> Path:
        return self.report_dir / self.report_top_rows_filename

    @property
    def report_missingness_by_column_path(self) -> Path:
        return self.report_dir / self.report_missingness_by_column_filename
