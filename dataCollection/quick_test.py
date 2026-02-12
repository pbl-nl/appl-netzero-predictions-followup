"""Quick test to verify imports and basic functionality before full test."""

import sys
from pathlib import Path

# Add parent directory to path to allow running as script
sys.path.append(str(Path(__file__).resolve().parent.parent))

print("DataCollection - Quick Import Test")
print("=" * 70)

# Test core imports
print("\n[1/7] Testing core imports...")
try:
    from dataCollection.core.base_collector import BaseCollector
    from dataCollection.core.protocols import RemoteResourceInfo, CollectionResult
    from dataCollection.config.base import BaseConfig, HTTPSettings
    print("  ✓ Core imports successful")
except Exception as e:
    print(f"  ✗ Core imports failed: {e}")
    exit(1)

# Test utilities
print("\n[2/7] Testing utility imports...")
try:
    from dataCollection.utils.http import HTTPClient
    from dataCollection.utils.metadata import MetadataManager
    from dataCollection.utils.hashing import sha256_file
    from dataCollection.utils.caching import HTTPConditionalGetStrategy, ChecksumStrategy
    print("  ✓ Utility imports successful")
except Exception as e:
    print(f"  ✗ Utility imports failed: {e}")
    exit(1)

# Test INFORM collector
print("\n[3/7] Testing INFORM collector...")
try:
    from dataCollection.collectors.inform import InformCollector
    from dataCollection.config.inform import InformConfig

    config = InformConfig(base_output_dir=Path("test_output"))
    print(f"  ✓ INFORM imports successful")
    print(f"    - Config: {config.source_name}")
    print(f"    - Output: {config.output_csv_path}")
except Exception as e:
    print(f"  ✗ INFORM failed: {e}")

# Test RSF collector
print("\n[4/7] Testing RSF collector...")
try:
    from dataCollection.collectors.rsf import RSFCollector
    from dataCollection.config.rsf import RSFConfig
    print("  ✓ RSF imports successful")
except Exception as e:
    print(f"  ✗ RSF failed: {e}")

# Test VDem collector
print("\n[5/7] Testing VDem collector...")
try:
    from dataCollection.collectors.vdem import VDemCollector
    from dataCollection.config.vdem import VDemConfig
    from dataCollection.utils.rdata import load_vdem_dataframe
    from dataCollection.utils.archive import extract_zip_to_temp
    print("  ✓ VDem imports successful")
except ImportError as e:
    print(f"  ⚠ VDem imports failed (pyreadr not installed?): {e}")
    print("    Install with: pip install pyreadr")
except Exception as e:
    print(f"  ✗ VDem failed: {e}")

# Test WorldBank collector
print("\n[6/7] Testing WorldBank collector...")
try:
    from dataCollection.collectors.worldbank import WorldBankCollector
    from dataCollection.config.worldbank import WorldBankConfig, WorldBankIndicator
    from dataCollection.utils.caching import WorldBankChangeDetectionStrategy
    print("  ✓ WorldBank imports successful")
except Exception as e:
    print(f"  ✗ WorldBank failed: {e}")

# Test PRIMAP collector
print("\n[7/7] Testing PRIMAP collector...")
try:
    from dataCollection.collectors.primap import PRIMAPCollector
    from dataCollection.config.primap import PRIMAPConfig
    print("  ✓ PRIMAP imports successful")
except Exception as e:
    print(f"  ✗ PRIMAP failed: {e}")

print("\n" + "=" * 70)
print("Quick Test Complete!")
print("\nNext steps:")
print("  1. If you see ⚠ warnings, install missing dependencies:")
print("     pip install pyreadr")
