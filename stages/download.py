import csv
import io
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = "data/raw"
REQUEST_TIMEOUT = 120
_cpu = os.cpu_count() or 1
MAX_WORKERS = max(1, _cpu - 1)

WORLD_BANK_COLUMNS = [
    "indicator_id",
    "indicator_name",
    "country_id",
    "country_name",
    "countryiso3code",
    "date",
    "value",
    "unit",
    "obs_status",
    "decimal",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

STATIC_SOURCES = [
    {
        "name": "inform",
        "kind": "xlsx",
        "url": "https://drmkc.jrc.ec.europa.eu/inform-index/Portals/0/InfoRM/2025/INFORM2024_TREND_2015_2024_v70_ALL.xlsx",
        "filename": "inform.xlsx",
        "use_browser_agent": False,
    },
    {
        "name": "primap",
        "kind": "zenodo_latest",
        "query": 'title:"PRIMAP-hist" AND access_right:"open"',
        "keyword": "no_rounding",
        "filename": "primap.csv",
    },
    {
        "name": "vdem",
        "kind": "csv_from_zip",
        "url": "https://www.v-dem.net/media/datasets/V-Dem-CY-Core-v15_csv.zip",
        "filename": "vdem.csv",
    },
    {
        "name": "data360_protein_supply",
        "kind": "csv",
        "url": "https://data360files.worldbank.org/data360-data/data/FAO_FS/FAO_FS_21014.csv",
        "filename": "data360_protein_supply.csv",
    },
]

WORLD_BANK_SOURCES = [
    {"name": "world_bank_oil_rents", "indicator": "NY.GDP.PETR.RT.ZS", "source": 2},
    {"name": "world_bank_natural_gas_rents", "indicator": "NY.GDP.NGAS.RT.ZS", "source": 2},
    {"name": "world_bank_coal_rents", "indicator": "NY.GDP.COAL.RT.ZS", "source": 2},
    {"name": "world_bank_forest_area", "indicator": "AG.LND.FRST.ZS", "source": 2},
    {"name": "world_bank_agricultural_land", "indicator": "AG.LND.AGRI.ZS", "source": 2},
    {"name": "world_bank_population_density", "indicator": "EN.POP.DNST", "source": 2},
    {"name": "world_bank_gdp_per_capita", "indicator": "NY.GDP.PCAP.KD", "source": 2},
    {"name": "world_bank_total_population", "indicator": "SP.POP.TOTL", "source": 2},
    {"name": "world_bank_gdp_ppp", "indicator": "NY.GDP.MKTP.PP.KD", "source": 2},
    {"name": "world_bank_annual_deforestation", "indicator": "ER.FST.DFST.ZG", "source": 18},
    {"name": "world_bank_control_of_corruption", "indicator": "CC.EST", "source": 2},
    {"name": "world_bank_absence_of_violence_terrorism", "indicator": "PV.EST", "source": 2},
    {"name": "world_bank_regulatory_quality", "indicator": "RQ.EST", "source": 2},
    {"name": "world_bank_voice_and_accountability", "indicator": "VA.EST", "source": 2},
]


def build_sources():
    sources = list(STATIC_SOURCES)
    for year in range(2002, 2025):
        if year != 2011:  # Skip 2011 as it's missing on the RSF site
            sources.append(
                {
                    "name": f"rsf_{year}",
                    "kind": "csv",
                    "url": f"https://rsf.org/sites/default/files/import_classement/{year}.csv",
                    "filename": f"rsf_{year}.csv",
                }
            )
    return sources


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def get_response(url, use_browser_agent=False):
    """
    Fetch a URL.
    By default, uses standard urllib agent (bypasses Cloudflare block on Zenodo API).
    If use_browser_agent=True, spoofs a browser (useful for tricky static files).
    """
    headers = {}
    if use_browser_agent:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    request = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            status_code = getattr(response, "status", 200)
            if status_code != 200:
                raise RuntimeError(f"HTTP {status_code} for {url}")
            return response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code} for {url}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Failed to reach {url}: {error.reason}") from error


def save_bytes(path, content):
    with open(path, "wb") as file:
        file.write(content)


def download_binary_file(source):
    content = get_response(source["url"], use_browser_agent=source.get("use_browser_agent", True))
    path = os.path.join(DATA_DIR, source["filename"])
    save_bytes(path, content)
    logging.info("Downloaded %s -> %s", source["name"], path)
    return path


def download_zenodo_latest(source):
    """Dynamically fetch the latest file URL from Zenodo."""
    
    # 1. URL-encode the advanced query so the API understands it
    safe_query = urllib.parse.quote(source["query"])
    api_url = f"https://zenodo.org/api/records?q={safe_query}&sort=mostrecent&size=1"
    
    # 2. Zenodo API blocks spoofed browser user-agents, so use default python urllib agent
    content = get_response(api_url, use_browser_agent=False)
    data = json.loads(content.decode("utf-8"))
    
    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        raise RuntimeError(f"No Zenodo records found for query '{source['query']}'")
        
    record = hits[0]
    download_url = None
    
    # 3. Scan the files in this official record for 'no_rounding' CSV
    for f in record.get("files", []):
        if f["key"].endswith(".csv") and source["keyword"] in f["key"].lower():
            download_url = f["links"]["self"]
            break
            
    if not download_url:
        raise RuntimeError(f"Could not find a CSV matching '{source['keyword']}' in Zenodo record")
        
    logging.info("Resolved %s Zenodo URL -> %s", source["name"], download_url)
    
    # 4. Download massive binary (also using default agent)
    file_content = get_response(download_url, use_browser_agent=False)
    
    path = os.path.join(DATA_DIR, source["filename"])
    save_bytes(path, file_content)
    logging.info("Downloaded %s -> %s", source["name"], path)
    return path


def download_csv_from_zip(source):
    content = get_response(source["url"], use_browser_agent=True)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_entries = [entry for entry in archive.infolist() if entry.filename.lower().endswith(".csv")]
        if not csv_entries:
            raise RuntimeError(f"No CSV file found inside zip for {source['name']}")
        largest_csv = max(csv_entries, key=lambda entry: entry.file_size)
        csv_bytes = archive.read(largest_csv.filename)

    path = os.path.join(DATA_DIR, source["filename"])
    save_bytes(path, csv_bytes)
    logging.info("Downloaded %s -> %s (from %s)", source["name"], path, largest_csv.filename)
    return path


def world_bank_url(indicator, source_id):
    return (
        f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
        f"?format=json&per_page=20000&page=1&source={source_id}"
    )


def parse_world_bank_payload(payload):
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError("World Bank response did not return [metadata, rows]")
    metadata = payload[0] if isinstance(payload[0], dict) else {}
    rows = payload[1] if isinstance(payload[1], list) else []
    return metadata, rows


def flatten_world_bank_row(row):
    indicator = row.get("indicator", {}) if isinstance(row, dict) else {}
    country = row.get("country", {}) if isinstance(row, dict) else {}
    return {
        "indicator_id": indicator.get("id"),
        "indicator_name": indicator.get("value"),
        "country_id": country.get("id"),
        "country_name": country.get("value"),
        "countryiso3code": (row.get("countryiso3code") or country.get("id")) if isinstance(row, dict) else None,
        "date": row.get("date") if isinstance(row, dict) else None,
        "value": row.get("value") if isinstance(row, dict) else None,
        "unit": row.get("unit") if isinstance(row, dict) else None,
        "obs_status": row.get("obs_status") if isinstance(row, dict) else None,
        "decimal": row.get("decimal") if isinstance(row, dict) else None,
    }


def write_world_bank_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=WORLD_BANK_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_world_bank_row(row))


def download_world_bank_csv(source):
    url = world_bank_url(source["indicator"], source["source"])
    content = get_response(url, use_browser_agent=False)
    payload = json.loads(content.decode("utf-8"))
    _, rows = parse_world_bank_payload(payload)
    path = os.path.join(DATA_DIR, f"{source['name']}.csv")
    write_world_bank_csv(path, rows)
    logging.info("Downloaded %s -> %s (rows=%s)", source["name"], path, len(rows))
    return path


def _dispatch_source(source):
    if source["kind"] == "csv_from_zip":
        download_csv_from_zip(source)
    elif source["kind"] == "zenodo_latest":
        download_zenodo_latest(source)
    else:
        download_binary_file(source)


def download_all(static_sources, world_bank_sources):
    tasks = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for source in static_sources:
            tasks[executor.submit(_dispatch_source, source)] = source["name"]
        for source in world_bank_sources:
            tasks[executor.submit(download_world_bank_csv, source)] = source["name"]

        failed_downloads = []
        for future in as_completed(tasks):
            name = tasks[future]
            try:
                future.result()
            except Exception as error:
                failed_downloads.append((name, str(error)))
                logging.error("Failed %s: %s", name, error)

    if failed_downloads:
        logging.error("Download failed for %s source(s).", len(failed_downloads))
        for name, reason in failed_downloads:
            logging.error("  %s -> %s", name, reason)
    else:
        logging.info("All sources downloaded successfully.")

    return failed_downloads


def run():
    ensure_data_dir()
    static_sources = build_sources()
    world_bank_sources = [dict(source) for source in WORLD_BANK_SOURCES]

    failed_downloads = download_all(static_sources, world_bank_sources)

    if failed_downloads:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
