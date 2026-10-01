"""Download selected DKASC Alice Springs arrays (full 5-min series with weather).

Terms (dkasolarcentre.com.au/download/terms-conditions, accepted by the user
on 2026-09-28): research use with citation and the DKA disclaimer; no
redistribution of more than 5,000 data cells without formal consent — raw
DKASC files therefore never enter the public reproducibility package; this
script is shipped instead.

Selection (before any data was seen): fixed-mount arrays commissioned
2008-2010 and still operating, one per distinct cell technology where
possible, for the forecasting screen and the long-horizon degradation module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/external/dkasc"
BASE = "https://solarcentre.spinifexvalley.com.au/export/"
ARRAYS = {
    "7":  ("79-Site_DKA-M6_A-Phase.csv", "First Solar, 7.0kW, CdTe, Fixed, 2008", 7.0),
    "8":  ("93-Site_DKA-M4_A-Phase.csv", "Kaneka, 6.0kW, Amorphous Silicon, Fixed, 2008", 6.0),
    "10": ("85-Site_DKA-M7_A-Phase.csv", "SunPower, 5.8kW, mono-Si, Fixed, 2009, 215W", 5.8),
    "11": ("106-Site_DKA-M5_C-Phase.csv", "BP Solar, 5.0kW, poly-Si, Fixed, 2008", 5.0),
    "13": ("92-Site_DKA-M6_B-Phase.csv", "Trina, 5.3kW, mono-Si, Fixed, 2009", 5.3),
    "17": ("69-Site_DKA-M4_B-Phase.csv", "Sanyo, 6.3kW, HIT Hybrid Silicon, Fixed, 2010", 6.3),
    # Added 2026-09-28 for the degradation paper (fixed, commissioned 2008-2011, still operating).
    "12": ("84-Site_DKA-M5_B-Phase.csv", "BP Solar, 5.1kW, mono-Si, Fixed, 2008", 5.1),
    "14": ("90-Site_DKA-M3_A-Phase.csv", "Kyocera, 5.4kW, poly-Si, Fixed, 2008", 5.4),
    "18": ("71-Site_DKA-M2_C-Phase.csv", "SunPower, 5.2kW, mono-Si, Fixed, 2011", 5.2),
    "19": ("98-Site_DKA-M8_B-Phase.csv", "Sungrid, 5.0kW, mono-Si, Fixed, 2010", 5.0),
    "20": ("68-Site_DKA-M8_C-Phase.csv", "Sungrid, 5.0kW, poly-Si, Fixed, 2010", 5.0),
    "21": ("67-Site_DKA-M8_A-Phase.csv", "Evergreen Solar, 4.9kW, poly-Si, Fixed, 2010", 4.9),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for array, (name, label, kw) in ARRAYS.items():
        target = OUT / name
        if target.exists() and name in manifest:
            continue
        digest = hashlib.sha256()
        part = target.with_suffix(".part")
        with requests.get(BASE + name, stream=True, timeout=300,
                          headers={"User-Agent": "research-data-download"}) as response:
            response.raise_for_status()
            with open(part, "wb") as handle:
                for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
        part.replace(target)
        manifest[name] = {"array": array, "label": label, "dc_kw": kw, "size": target.stat().st_size,
                          "sha256": digest.hexdigest(), "source": BASE + name,
                          "terms": "DKASC terms; no raw redistribution > 5,000 cells"}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(manifest[name], flush=True)


if __name__ == "__main__":
    main()
