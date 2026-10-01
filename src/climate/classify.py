"""Sample a documented 1991–2020 Köppen–Geiger raster at plant locations."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform

# Beck et al. (2023), Scientific Data, doi:10.1038/s41597-023-02549-6.
# Numeric class mapping must be supplied from the exact downloaded raster's
# accompanying legend. Never infer class names from colour or nearby cities.


def load_codebook(legend_path: Path) -> dict[int, str]:
    """Read numeric climate codes from the matching archive's legend.txt."""
    codebook: dict[int, str] = {}
    pattern = re.compile(r"^\s*(\d+):\s+([A-Z][A-Za-z]{1,2})\b")
    for line in Path(legend_path).read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            code, symbol = int(match.group(1)), match.group(2)
            if code in codebook:
                raise ValueError(f"Duplicate climate code {code}")
            codebook[code] = symbol
    if set(codebook) != set(range(1, 31)) or len(set(codebook.values())) != 30:
        raise ValueError("Expected 30 unique Köppen classes numbered 1–30")
    return codebook


def verify_reference(manifest_path: Path) -> tuple[Path, dict[int, str], dict]:
    """Verify exact archived files and the documented historical raster grid."""
    manifest_path = Path(manifest_path)
    info = json.loads(manifest_path.read_text(encoding="utf-8"))
    folder = manifest_path.parent
    raster_path = folder / info["raster_file"]
    legend_path = folder / info["legend_file"]
    for path, key in ((raster_path, "raster_sha256"),
                      (legend_path, "legend_sha256")):
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual.lower() != info[key].lower():
            raise ValueError(f"Checksum mismatch: {path.name}")
    codebook = load_codebook(legend_path)
    with rasterio.open(raster_path) as raster:
        if (str(raster.crs) != "EPSG:4326" or raster.width != 43200 or
                raster.height != 21600 or raster.count != 1 or
                raster.dtypes[0] != "uint8" or raster.nodata != 0 or
                not np.allclose(raster.res, (1 / 120, 1 / 120)) or
                not np.allclose(tuple(raster.bounds), (-180, -90, 180, 90))):
            raise ValueError("Historical 0.00833333° climate raster metadata mismatch")
    return raster_path, codebook, info


def classify_sites(registry: pd.DataFrame, raster_path: Path,
                   codebook: dict[int, str]) -> pd.DataFrame:
    required = {"latitude", "longitude", "system_id"}
    if not required.issubset(registry):
        raise ValueError(f"Missing columns: {required - set(registry)}")
    if not codebook:
        raise ValueError("Exact raster codebook is required")
    output = registry.copy()
    with rasterio.open(raster_path) as raster:
        if raster.crs is None:
            raise ValueError("Climate raster has no CRS")
        coords = []
        for lat, lon in zip(output.latitude, output.longitude):
            if pd.isna(lat) or pd.isna(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
                coords.append((float("nan"), float("nan")))
            else:
                x, y = transform("EPSG:4326", raster.crs, [float(lon)], [float(lat)])
                coords.append((x[0], y[0]))
        codes = []
        for xy in coords:
            if any(pd.isna(value) for value in xy):
                codes.append(None)
            else:
                value = next(raster.sample([xy], masked=True))[0]
                codes.append(None if np.ma.is_masked(value) or pd.isna(value)
                             else int(value))
    unknown_codes = {value for value in codes if value is not None and value not in codebook}
    if unknown_codes:
        raise ValueError(f"Climate codes absent from codebook: {sorted(unknown_codes)}")
    output["climate_code"] = codes
    output["climate_class"] = [codebook.get(code) for code in codes]
    output["climate_family"] = output.climate_class.str[:1]
    output["climate_source"] = "Beck et al. 2023, 1991-2020, doi:10.1038/s41597-023-02549-6"
    return output
