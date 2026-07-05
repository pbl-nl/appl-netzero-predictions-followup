from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_NETZERO = ROOT / "data" / "netzero"

# Host project's own data dir — the only two files simplified reads from
# outside its own tree (IEA csv, NZ workbook) live here.
PARENT_ROOT = ROOT.parent
PARENT_DATA_RAW = PARENT_ROOT / "data" / "raw"
PARENT_DATA_NETZERO = PARENT_ROOT / "data" / "netzero"

YEARS = list(range(2000, 2025))
YEAR_COLS = [str(y) for y in YEARS]

# World Bank indicators: (name, indicator_code, source_id)
WB_INDICATORS = [
    ("oil_rents",           "NY.GDP.PETR.RT.ZS", 2),
    ("gas_rents",           "NY.GDP.NGAS.RT.ZS",  2),
    ("coal_rents",          "NY.GDP.COAL.RT.ZS",  2),
    ("forest_area",         "AG.LND.FRST.ZS",     2),
    ("agricultural_land",   "AG.LND.AGRI.ZS",     2),
    ("population_density",  "EN.POP.DNST",         2),
    ("gdp_per_capita",      "NY.GDP.PCAP.KD",      2),
    ("population",          "SP.POP.TOTL",         2),
    ("gdp_ppp",             "NY.GDP.MKTP.PP.KD",   2),
    ("deforestation",       "ER.FST.DFST.ZG",      18),
]

WB_API_URL = (
    "https://api.worldbank.org/v2/country/all/indicator/{indicator}"
    "?format=json&per_page=20000&page=1&source={source_id}"
)

# WGI sheets and indicator labels
WGI_SHEETS = {
    "cc": "control_of_corruption",
    "pv": "political_stability",
    "rq": "regulatory_quality",
    "va": "voice_accountability",
}
WGI_URL = (
    "https://datacatalogfiles.worldbank.org/ddh-published/0038026/DR0095947/"
    "wgidataset_with_sourcedata-2025.xlsx"
)

INFORM_URL = (
    "https://drmkc.jrc.ec.europa.eu/inform-index/Portals/0/InfoRM/2025/"
    "INFORM2024_TREND_2015_2024_v70_ALL.xlsx"
)

VDEM_URL = "https://www.v-dem.net/media/datasets/V-Dem-CY-Core-v15_csv.zip"

FAO_URL = (
    "https://data360files.worldbank.org/data360-data/data/FAO_FS/FAO_FS_21014.csv"
)

UN_WPP_URL = (
    "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/"
    "EXCEL_FILES/1_General/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_COMPACT.xlsx"
)

RSF_URL_TEMPLATE = (
    "https://rsf.org/sites/default/files/import_classement/{year}.csv"
)
RSF_YEARS = [y for y in range(2002, 2025) if y != 2011]

RENEWABLE_PRODUCTS = {
    "HYDRO", "WIND", "SOLARPV", "SOLARTH", "GEOTHERM", "TIDE",
    "PRIMSBIO", "BIOGASES", "BIOGASOL", "BIODIESEL", "OBIOLIQ",
    "RENEWNS", "MUNWASTER", "CHARCOAL",
}

NZ_EXCEL_FILENAME = "Reduction rates NZ target revised_18May2026.xlsx"
IEA_FILENAME = "8624f431-en_2023.csv"
