"""Parse a per-sample reduction config JSON into ReductionMeta.

Observed schema (drtsans/eqsanscli): sample.runNumber, sample.thickness,
sample.transmission.runNumber, background.runNumber,
background.transmission.runNumber, emptyTransmission.runNumber,
beamCenter.runNumber, outputFileName, iptsNumber, and a configuration block
with maskFileName, StandardAbsoluteScale, absoluteScaleMethod, Qmin, Qmax,
numQBins. All keys read defensively.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..models import ReductionMeta

logger = logging.getLogger(__name__)


def _get(d: dict, *path):
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse(path: Path) -> ReductionMeta:
    try:
        raw = json.loads(Path(path).read_text())
    except Exception as e:
        logger.warning("Could not parse reduction JSON %s: %s", path, e)
        return ReductionMeta(output_name=Path(path).stem, source_json=Path(path))

    cfg = raw.get("configuration", {}) if isinstance(raw, dict) else {}
    return ReductionMeta(
        output_name=raw.get("outputFileName") or Path(path).stem,
        ipts=_to_int(raw.get("iptsNumber")),
        sample_run=_str(_get(raw, "sample", "runNumber")),
        sample_thickness=_to_float(_get(raw, "sample", "thickness")),
        trans_run=_str(_get(raw, "sample", "transmission", "runNumber")),
        bkg_run=_str(_get(raw, "background", "runNumber")),
        bkg_trans_run=_str(_get(raw, "background", "transmission", "runNumber")),
        empty_trans_run=_str(_get(raw, "emptyTransmission", "runNumber")),
        beam_center_run=_str(_get(raw, "beamCenter", "runNumber")),
        mask_file=_str(cfg.get("maskFileName")),
        qmin=_to_float(cfg.get("Qmin")),
        qmax=_to_float(cfg.get("Qmax")),
        num_q_bins=_to_int(cfg.get("numQBins")),
        abs_scale=_to_float(cfg.get("StandardAbsoluteScale")),
        abs_scale_method=_str(cfg.get("absoluteScaleMethod")),
        source_json=Path(path),
        config_raw=cfg if isinstance(cfg, dict) else {},
    )


def _str(v):
    if v is None:
        return None
    s = str(v).strip()
    return s or None
