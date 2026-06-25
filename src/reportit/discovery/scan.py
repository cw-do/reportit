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
    merged = [p for n, p in files.items()
              if n.lower().startswith("merged_") and n.lower().endswith(".txt")]
    # index merged files by (base, temperature) — a merged file combines the
    # configurations for one sample+condition, so both configs map to it.
    merged_index: dict[tuple, Path] = {}
    for p in merged:
        b, t = _parse_merged(p.name)
        if b is not None:
            merged_index.setdefault((b, t), p)

    for name, path in sorted(files.items()):
        m = _IQ1D_RE.match(name)
        if not m:
            continue
        out_name = m.group("name")

        iqxqy = output_dir / f"{out_name}_Iqxqy.dat"
        json_path = output_dir / f"{out_name}.json"
        trans_path = output_dir / f"{out_name}_trans.txt"

        base, temp, config = naming.parse_sample_name(out_name)
        merged_path = merged_index.get((base, temp))

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


_CONFIG_TOKEN = re.compile(r"^\d+(?:\.\d+)?m\d+(?:\.\d+)?a(?:\d+hz)?$", re.IGNORECASE)
_TEMP_TOKEN = re.compile(r"^-?\d+(?:\.\d+)?C$", re.IGNORECASE)


def _parse_merged(filename: str) -> tuple[str | None, str | None]:
    """Parse 'merged_<base>[_<temp>]_<cfg>_<cfg>..._Iq.txt' -> (base, temp).

    e.g. merged_15_30C_4m10a_2.5m2.5a_Iq.txt -> ('15', '30C')
         merged_15_4m10a_2.5m2.5a_Iq.txt     -> ('15', None)
    """
    m = re.match(r"^merged_(?P<core>.+?)_Iq\.\w+$", filename, re.IGNORECASE)
    if not m:
        return None, None
    tokens = m.group("core").split("_")
    while tokens and _CONFIG_TOKEN.match(tokens[-1]):  # strip trailing config combo
        tokens.pop()
    temp = None
    if tokens and _TEMP_TOKEN.match(tokens[-1]):
        temp = tokens.pop()
    base = "_".join(tokens) if tokens else None
    return base, temp
