"""Builders for the report's tables.

- Sample Summary (main body): sample, description, configurations, conditions.
  Deliberately NO single run number (that was confusing when a row spanned
  several temperatures/configs).
- Reduction Run Table (appendix): per sample+condition, the scattering and
  transmission run for each configuration plus background/empty — laid out so a
  reduction table can be rebuilt by hand.
- Run Catalog (appendix): raw ONCat info (run, title, distance, wavelength,
  counts, duration).
"""

from __future__ import annotations

import re

from ..models import AnalysisStrategy, Dataset, TableSpec


def _natkey(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _temp_val(t: str | None):
    if not t:
        return -1.0
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    return float(m.group()) if m else -1.0


def _science(datasets: list[Dataset], strategy: AnalysisStrategy) -> list[Dataset]:
    variants = set(strategy.variant_decision.variants_used or [])
    out = []
    for d in datasets:
        if d.is_standard:
            continue
        if variants and d.variant not in variants:
            continue
        out.append(d)
    # if variant filter removed everything, fall back to all non-standard
    return out or [d for d in datasets if not d.is_standard]


def _describe(base: str, rep: Dataset | None, proposal) -> str:
    if proposal and proposal.sample_descriptions:
        descs = proposal.sample_descriptions
        if base in descs:
            return descs[base]
        bl = base.lower()
        for k, v in descs.items():
            kl = str(k).lower()
            if kl == bl or bl in kl or kl in bl:
                return v
    if rep and rep.oncat_title:
        return rep.oncat_title
    return "—"


def _drop_empty_columns(headers, rows, colspec, *, keep: int = 1):
    """Remove columns that are empty ('—') in every row.

    Merged-only datasets carry no reduction JSON, so Description / Configurations
    / Conditions can be entirely unknown — a column of dashes is noise, and a
    table of nothing but names says less than no table at all.
    """
    specs = colspec.split() if colspec else [""] * len(headers)
    if len(specs) != len(headers):
        specs = [""] * len(headers)
    keep_idx = [i for i in range(len(headers))
                if i < keep or any(str(r[i]).strip() not in ("—", "", "-") for r in rows)]
    if len(keep_idx) == len(headers):
        return headers, rows, colspec, False
    headers = [headers[i] for i in keep_idx]
    rows = [[r[i] for i in keep_idx] for r in rows]
    colspec = " ".join(specs[i] for i in keep_idx).strip()
    return headers, rows, colspec, True


def build_sample_summary(datasets, strategy, proposal, namemap=None) -> TableSpec | None:
    science = _science(datasets, strategy)
    short = namemap.short if namemap is not None else (lambda x: x)
    by_base: dict[str, dict] = {}
    for d in science:
        rec = by_base.setdefault(d.base, {"configs": set(), "temps": set(), "rep": None})
        if d.config:
            rec["configs"].add(d.config)
        if d.temperature:
            rec["temps"].add(d.temperature)
        if rec["rep"] is None and d.meta and d.meta.sample_run:
            rec["rep"] = d
    if not by_base:
        return None

    rows = []
    for base in sorted(by_base, key=_natkey):
        rec = by_base[base]
        desc = _describe(base, rec["rep"], proposal)
        configs = ", ".join(sorted(rec["configs"], key=_natkey)) or "—"
        temps = ", ".join(sorted(rec["temps"], key=_temp_val)) or "—"
        rows.append([short(base), desc, configs, temps])

    headers = ["Sample", "Description", "Configurations", "Conditions"]
    colspec = "l p{6.2cm} p{3.2cm} p{3.0cm}"
    headers, rows, colspec, trimmed = _drop_empty_columns(headers, rows, colspec)
    if len(headers) <= 1:
        # Only the sample names survived — nothing this table can tell the reader
        # that the group sections and the ONCat catalog do not already say.
        return None

    caption = ("Samples measured in this experiment (calibration standards excluded). "
               "Conditions list the temperatures measured for each sample.")
    if trimmed:
        caption += (" Columns with no information for this experiment (the reduced "
                    "files carry no reduction metadata) are omitted.")
    return TableSpec(
        caption=caption,
        label="tab:samples",
        headers=headers,
        rows=rows,
        longtable=True, fontsize="small",
        colspec=colspec,
    )


def build_name_map_table(namemap) -> TableSpec | None:
    """Legend for the short names used throughout the report."""
    if namemap is None or not namemap.active:
        return None
    pairs = namemap.pairs()
    if not pairs:
        return None
    return TableSpec(
        caption="Short names used in the tables, figures and fits of this report, "
                "and the reduced-output file each one refers to. Short names are "
                "derived by dropping the leading and trailing name parts that every "
                "file shares, so only the distinguishing part is shown.",
        label="tab:shortnames",
        headers=["Short name", "Full reduced-output name"],
        rows=[[s, f] for s, f in pairs],
        longtable=True, fontsize="footnotesize",
        colspec="l p{12cm}",
        section_title="Short Name Legend",
    )


def build_reduction_table(datasets, strategy, namemap=None) -> TableSpec | None:
    science = _science(datasets, strategy)
    short = namemap.short if namemap is not None else (lambda x: x)
    configs = sorted({d.config for d in science if d.config}, key=_natkey)
    if not configs:
        return None

    # group by (base, temperature)
    keyed: dict[tuple, dict] = {}
    for d in science:
        k = (d.base, d.temperature or "")
        rec = keyed.setdefault(k, {"by_config": {}, "bkg": None, "empty": None})
        m = d.meta
        if m:
            rec["by_config"][d.config] = (m.sample_run, m.trans_run)
            if rec["bkg"] is None and m.bkg_run:
                rec["bkg"] = m.bkg_run
            if rec["empty"] is None and (m.empty_trans_run or m.bkg_trans_run):
                rec["empty"] = m.empty_trans_run or m.bkg_trans_run

    headers = ["Sample", "Cond."]
    for c in configs:
        headers += [f"{c} scatt", f"{c} trans"]
    headers += ["Bkg", "Empty"]

    rows = []
    for (base, temp) in sorted(keyed, key=lambda kt: (_natkey(kt[0]), _temp_val(kt[1]))):
        rec = keyed[(base, temp)]
        row = [short(base), temp or "—"]
        for c in configs:
            scatt, trans = rec["by_config"].get(c, (None, None))
            row += [scatt or "—", trans or "—"]
        row += [rec["bkg"] or "—", rec["empty"] or "—"]
        rows.append(row)

    wide = len(headers) >= 7
    colspec = "ll" + "ll" * len(configs) + "ll"
    return TableSpec(
        caption="Run numbers per sample and condition: scattering and transmission "
                "for each configuration, plus background and empty-cell transmission. "
                "Use this to reconstruct a reduction table.",
        label="tab:runtable",
        headers=headers, rows=rows,
        longtable=True, landscape=wide, fontsize="footnotesize", colspec=colspec,
        section_title="Reduction Run Table",
    )


def build_catalog_table(catalog) -> TableSpec | None:
    if catalog is None or getattr(catalog, "empty", True):
        return None
    rows = []
    try:
        for _, r in catalog.sort_values("run_number").iterrows():
            rows.append([
                str(int(r["run_number"])),
                (str(r.get("title", "")) or "—")[:80],
                f"{float(r.get('detector_distance', 0)):.2f}",
                f"{float(r.get('wavelength', 0)):.2f}",
                f"{int(r.get('total_counts', 0)):,}",
                str(int(r.get("duration", 0))),
            ])
    except Exception:  # noqa: BLE001
        return None
    if not rows:
        return None
    return TableSpec(
        caption="ONCat run catalog for this IPTS: run number, title, sample-to-detector "
                "distance, wavelength, total counts, and duration.",
        label="tab:catalog",
        headers=["Run", "Title", "Dist (m)", "Wavelength (A)", "Counts", "Dur (s)"],
        rows=rows,
        longtable=True, fontsize="footnotesize",
        colspec="l p{6.2cm} r r r r",
        section_title="ONCat Run Catalog",
    )
