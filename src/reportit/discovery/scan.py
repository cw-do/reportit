"""Scan candidate output directories into Dataset records.

A Dataset = one output_name within one variant dir, with its 1D/2D/merged/json
siblings attached. The variant label is the output dir name (e.g. "output",
"output_mask4") so multiple reductions of the same samples stay distinguishable.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ..models import Dataset
from . import naming, reduction_json

logger = logging.getLogger(__name__)

_IQ1D_RE = re.compile(r"^(?P<name>.+)_Iq\.dat$", re.IGNORECASE)


def scan_dir(output_dir: Path, variant: str | None = None) -> list[Dataset]:
    output_dir = Path(output_dir)
    variant = variant or output_dir.name
    datasets: list[Dataset] = []

    files = {p.name: p for p in output_dir.iterdir() if p.is_file()}
    merged = [p for n, p in files.items() if n.lower().startswith("merged_") and n.lower().endswith(".txt")]

    for name, path in sorted(files.items()):
        m = _IQ1D_RE.match(name)
        if not m:
            continue
        out_name = m.group("name")

        iqxqy = output_dir / f"{out_name}_Iqxqy.dat"
        json_path = output_dir / f"{out_name}.json"
        trans_path = output_dir / f"{out_name}_trans.txt"
        merged_path = _find_merged(out_name, merged)

        base, temp, config = naming.parse_sample_name(out_name)
        meta = reduction_json.parse(json_path) if json_path.is_file() else None

        datasets.append(Dataset(
            output_name=out_name,
            variant=variant,
            base=base,
            temperature=temp,
            config=config,
            iq_path=path,
            iqxqy_path=iqxqy if iqxqy.is_file() else None,
            merged_path=merged_path,
            trans_path=trans_path if trans_path.is_file() else None,
            meta=meta,
            is_standard=naming.is_standard(base),
        ))

    logger.info("Scanned %d datasets in %s", len(datasets), output_dir)
    return datasets


def _find_merged(out_name: str, merged: list[Path]) -> Path | None:
    """Match a merged_* file that contains this output name (sans config tail)."""
    base, temp, _ = naming.parse_sample_name(out_name)
    needle = f"{base}_{temp}" if temp else base
    for p in merged:
        if out_name in p.name or needle in p.name:
            return p
    return None
