"""ONCat API wrapper — fetches EQSANS run catalog via pyoncat.

Copied/adapted from eqsanstools-cli integrations/oncat.py. Machine-to-machine
client-credentials flow; no interactive login required. Results cached to disk.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

CLIENT_ID = "17ddcb3e-a727-41a2-aec5-43533988ab69"
CLIENT_SECRET = "3027a2b1-da09-4e13-bf97-f389ff1a747f"

PROJECTION = [
    "experiment",
    "location",
    "indexed.run_number",
    "metadata.entry.title",
    "metadata.entry.run_number",
    "metadata.entry.total_counts",
    "metadata.entry.duration",
    "metadata.entry.daslogs.detectorz.average_value",
    "metadata.entry.daslogs.wavelength.average_value",
    "metadata.entry.daslogs.speed1.average_value",
    "metadata.entry.proton_charge",
]


def _round_frequency(raw_freq: float) -> int:
    if raw_freq <= 0:
        return 60
    return 30 if raw_freq < 45 else 60


def _extract_field(record: Any, dotted_path: str) -> Any:
    obj = record
    for key in dotted_path.split("."):
        if obj is None:
            return None
        try:
            obj = obj[key]
        except (KeyError, TypeError, IndexError):
            try:
                obj = getattr(obj, key, None)
            except Exception:
                return None
    return obj


def fetch_catalog(ipts: int) -> pd.DataFrame:
    """Fetch all runs for an IPTS number from ONCat.

    Columns: run_number, title, detector_distance, wavelength, total_counts,
    duration, frequency, proton_charge, experiment, location.
    """
    try:
        import pyoncat
    except ImportError as e:
        raise ImportError("pyoncat is required for ONCat access.") from e

    logger.info("Connecting to ONCat for IPTS-%d...", ipts)
    oncat = pyoncat.ONCat(
        "https://oncat.ornl.gov",
        flow=pyoncat.CLIENT_CREDENTIALS_FLOW,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )
    oncat.login()

    datafiles = oncat.Datafile.list(
        facility="SNS",
        instrument="EQSANS",
        experiment=f"IPTS-{ipts}",
        projection=PROJECTION,
        exts=[".nxs.h5"],
    )
    if not datafiles:
        logger.warning("No datafiles found for IPTS-%d", ipts)
        return pd.DataFrame()

    rows = []
    for record in datafiles:
        run_number = (_extract_field(record, "metadata.entry.run_number")
                      or _extract_field(record, "indexed.run_number"))
        if run_number is None:
            continue
        rows.append({
            "run_number": int(run_number),
            "title": _extract_field(record, "metadata.entry.title") or "",
            "detector_distance": float(
                _extract_field(record, "metadata.entry.daslogs.detectorz.average_value") or 0
            ) / 1000.0,
            "wavelength": float(
                _extract_field(record, "metadata.entry.daslogs.wavelength.average_value") or 0
            ),
            "total_counts": int(_extract_field(record, "metadata.entry.total_counts") or 0),
            "duration": int(_extract_field(record, "metadata.entry.duration") or 0),
            "frequency": _round_frequency(float(
                _extract_field(record, "metadata.entry.daslogs.speed1.average_value") or 60
            )),
            "proton_charge": float(_extract_field(record, "metadata.entry.proton_charge") or 0),
            "experiment": _extract_field(record, "experiment") or f"IPTS-{ipts}",
            "location": _extract_field(record, "location") or "",
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("run_number").reset_index(drop=True)
    logger.info("Fetched %d runs for IPTS-%d", len(df), ipts)
    return df


def fetch_catalog_cached(ipts: int, cache, refresh: bool = False) -> Optional[pd.DataFrame]:
    """Cached wrapper. Returns None (and logs) if ONCat is unavailable."""
    key = f"oncat:catalog:{ipts}"
    if not refresh and cache is not None:
        hit = cache.get(key)
        if hit is not None:
            try:
                return pd.DataFrame(hit)
            except Exception:
                pass
    try:
        df = fetch_catalog(ipts)
    except Exception as e:
        hint = ""
        if "token" in str(e).lower():
            hint = (" — ONCat auth/server appears to be down (e.g. HTTP 502); this "
                    "is not a reportit or credentials problem. Proceeding with run "
                    "titles inferred from filenames; retry later with --refresh.")
        logger.warning("ONCat unavailable for IPTS-%d: %s%s", ipts, e, hint)
        return None
    if cache is not None and not df.empty:
        cache.set(key, df.to_dict(orient="records"))
    return df
