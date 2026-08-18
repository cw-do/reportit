"""ONCat API wrapper — fetches EQSANS run catalog via pyoncat.

Copied/adapted from eqsanstools-cli integrations/oncat.py. Machine-to-machine
client-credentials flow; no interactive login required. Results cached to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
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


def _oncat_client(ipts: int):
    """A logged-in ONCat client (machine-to-machine, no interactive login)."""
    try:
        import pyoncat
    except ImportError as e:
        raise ImportError("pyoncat is required for ONCat access.") from e
    logger.info("Connecting to ONCat for IPTS-%s...", ipts)
    oncat = pyoncat.ONCat(
        "https://oncat.ornl.gov",
        flow=pyoncat.CLIENT_CREDENTIALS_FLOW,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
    )
    oncat.login()
    return oncat


def fetch_proposal_pdf(ipts: int, dest: Path) -> Optional[Path]:
    """Download the beamtime proposal (statement of research) PDF from ONCat.

    The PDF lives in the proposal's ``statement_of_research`` field as a base64
    blob. Writes it to ``dest`` and returns the path, or None if unavailable
    (field missing, not a PDF, or ONCat down) — never raises, so a run continues
    without a proposal exactly as it would if none were on disk.
    """
    import base64

    try:
        oncat = _oncat_client(ipts)
        proposal = oncat.Proposal.retrieve(
            f"IPTS-{ipts}", projection=["statement_of_research"])
        blob = getattr(proposal, "statement_of_research", None)
        if not blob:
            logger.info("ONCat has no statement_of_research for IPTS-%s.", ipts)
            return None
        raw = base64.b64decode(blob)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not fetch proposal PDF from ONCat for IPTS-%s: %s",
                       ipts, e)
        return None
    if raw[:5] != b"%PDF-":
        logger.warning("ONCat statement_of_research for IPTS-%s is not a PDF "
                       "(starts with %r); ignoring.", ipts, raw[:8])
        return None
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
    except OSError as e:
        logger.warning("Could not write proposal PDF to %s: %s", dest, e)
        return None
    logger.info("Downloaded proposal PDF from ONCat for IPTS-%s (%d bytes) -> %s",
                ipts, len(raw), dest)
    return dest


def _proposal_backup_path(ipts: int) -> Path:
    """A stable, per-user location where the last good proposal download is kept,
    so a later ONCat outage can reuse it instead of falling back to nothing."""
    return Path.home() / ".reportit" / "proposals" / f"proposal_IPTS-{ipts}.pdf"


def _is_pdf(p: Path) -> bool:
    try:
        with open(p, "rb") as fh:
            return fh.read(5) == b"%PDF-"
    except OSError:
        return False


def fetch_proposal_pdf_cached(ipts: int, dest_dir: Path,
                              refresh: bool = False) -> Optional[Path]:
    """Get the proposal PDF into ``dest_dir`` (kept alongside the report, not
    hidden in the cache), returning its path or None.

    Robust against a flaky ONCat: every successful download is also copied to a
    stable per-user backup (``~/.reportit/proposals/``), and if a later download
    fails that backup is reused. So a transient ONCat outage no longer silently
    drops the proposal — and with it every proposal-driven analysis. ``refresh``
    forces a fresh download but still falls back to the backup if that fails.
    """
    import shutil

    dest = Path(dest_dir) / f"proposal_IPTS-{ipts}.pdf"
    backup = _proposal_backup_path(ipts)

    # already downloaded next to the report — reuse unless --refresh
    if not refresh and dest.is_file() and dest.stat().st_size > 0 and _is_pdf(dest):
        logger.info("Using previously downloaded proposal PDF %s", dest)
        return dest

    got = fetch_proposal_pdf(ipts, dest)
    if got is not None:
        try:  # keep a stable backup for next time ONCat is down
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(got, backup)
        except OSError as e:
            logger.debug("Could not update proposal backup %s: %s", backup, e)
        return got

    # download failed — fall back to the last good copy if we have one
    if backup.is_file() and backup.stat().st_size > 0 and _is_pdf(backup):
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, dest)
            logger.warning("ONCat proposal download failed for IPTS-%s; reusing the "
                           "last good copy from %s.", ipts, backup)
            return dest
        except OSError as e:
            logger.warning("Could not restore proposal backup %s: %s", backup, e)
    return None


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
