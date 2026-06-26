"""Read-only probe implementations the strategist LLM can call.

All file access is restricted to within the experiment's shared directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..analysis.loaders import downsample_curve
from ..discovery import reduction_json
from ..models import Dataset

logger = logging.getLogger(__name__)


class Probes:
    def __init__(self, shared_dir: Path, datasets: list[Dataset], catalog=None):
        self.shared_dir = Path(shared_dir).resolve()
        self.datasets = datasets
        self.catalog = catalog
        self._by_key = {(d.variant, d.output_name): d for d in datasets}
        # also index by output_name alone (first match) for convenience
        self._by_name: dict[str, Dataset] = {}
        for d in datasets:
            self._by_name.setdefault(d.output_name, d)

    # -- path safety ------------------------------------------------------ #
    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.shared_dir / p
        p = p.resolve()
        if self.shared_dir not in p.parents and p != self.shared_dir:
            raise ValueError(f"Path escapes shared dir: {path}")
        if self._in_output(p):
            raise ValueError(f"Path is inside a reportit output directory: {path}")
        return p

    def _in_output(self, p: Path) -> bool:
        """True if p is within a reportit-generated report dir (so the agent never
        reads its own prior output). Checks p and parents down to the shared dir."""
        from ..discovery.inventory import is_reportit_output_dir
        d = p
        while True:
            if d.is_dir() and is_reportit_output_dir(d):
                return True
            if d == self.shared_dir or self.shared_dir not in d.parents:
                return False
            d = d.parent

    # -- dispatch --------------------------------------------------------- #
    def dispatch(self, name: str, args: dict) -> Any:
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        return fn(args)

    # -- tools ------------------------------------------------------------ #
    def _t_list_dir(self, args: dict) -> Any:
        p = self._resolve(args["path"])
        if not p.is_dir():
            return {"error": f"not a directory: {p}"}
        from ..discovery.inventory import is_reportit_output_dir
        out = []
        for child in sorted(p.iterdir())[:200]:
            if child.is_dir() and is_reportit_output_dir(child):
                continue  # hide reportit's own output directories
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
            out.append({"name": child.name, "is_dir": child.is_dir(), "size": size})
        return {"path": str(p.relative_to(self.shared_dir)), "entries": out,
                "count": len(list(p.iterdir()))}

    def _t_read_text(self, args: dict) -> Any:
        p = self._resolve(args["path"])
        # generous default — these models have large context windows, so don't
        # truncate normal files (NOTE.md, scripts, JSONs are all well under this)
        max_bytes = int(args.get("max_bytes") or 120000)
        if not p.is_file():
            return {"error": f"not a file: {p}"}
        try:
            data = p.read_text(errors="replace")[:max_bytes]
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
        return {"path": str(p.relative_to(self.shared_dir)), "text": data}

    def _t_head_file(self, args: dict) -> Any:
        p = self._resolve(args["path"])
        n = int(args.get("n") or 60)
        if not p.is_file():
            return {"error": f"not a file: {p}"}
        lines = []
        with open(p, errors="replace") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                lines.append(line.rstrip("\n"))
        return {"path": str(p.relative_to(self.shared_dir)), "lines": lines}

    def _t_parse_reduction_json(self, args: dict) -> Any:
        p = self._resolve(args["path"])
        if not p.is_file():
            return {"error": f"not a file: {p}"}
        meta = reduction_json.parse(p)
        return {
            "output_name": meta.output_name, "ipts": meta.ipts,
            "sample_run": meta.sample_run, "sample_thickness": meta.sample_thickness,
            "trans_run": meta.trans_run, "bkg_run": meta.bkg_run,
            "empty_trans_run": meta.empty_trans_run, "beam_center_run": meta.beam_center_run,
            "mask_file": meta.mask_file, "qmin": meta.qmin, "qmax": meta.qmax,
            "num_q_bins": meta.num_q_bins, "abs_scale": meta.abs_scale,
            "abs_scale_method": meta.abs_scale_method,
        }

    def _t_oncat_titles(self, args: dict) -> Any:
        runs = [str(r).strip() for r in (args.get("runs") or [])]
        if self.catalog is None or getattr(self.catalog, "empty", True):
            return {"error": "ONCat catalog unavailable", "titles": {}}
        out = {}
        for r in runs:
            try:
                row = self.catalog[self.catalog["run_number"] == int(r)]
            except (ValueError, KeyError):
                row = None
            if row is not None and len(row):
                rec = row.iloc[0]
                out[r] = {"title": str(rec.get("title", "")),
                          "detector_distance_m": float(rec.get("detector_distance", 0)),
                          "wavelength": float(rec.get("wavelength", 0)),
                          "duration_s": int(rec.get("duration", 0))}
            else:
                out[r] = {"title": None}
        return {"titles": out}

    def _t_sample_curve(self, args: dict) -> Any:
        name = args.get("output_name")
        variant = args.get("variant")
        pts = int(args.get("points") or 25)
        ds = self._by_key.get((variant, name)) if variant else self._by_name.get(name)
        if ds is None or not ds.iq_path:
            return {"error": f"no 1D data for {name!r} (variant={variant!r})"}
        try:
            curve = downsample_curve(ds.iq_path, pts)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
        return {"output_name": name, "variant": ds.variant,
                "q_I": [[round(q, 6), round(i, 6)] for q, i in curve]}

    def _t_list_datasets(self, args: dict) -> Any:
        variant = args.get("variant")
        out = []
        for d in self.datasets:
            if variant and d.variant != variant:
                continue
            out.append({"output_name": d.output_name, "variant": d.variant,
                        "base": d.base, "temperature": d.temperature,
                        "config": d.config, "is_standard": d.is_standard,
                        "sample_run": d.meta.sample_run if d.meta else None,
                        "has_2d": d.iqxqy_path is not None,
                        "has_merged": d.merged_path is not None})
        return {"count": len(out), "datasets": out}
