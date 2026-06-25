"""Dataclasses passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Discovery / inventory
# --------------------------------------------------------------------------- #
@dataclass
class FileEntry:
    path: Path
    size: int
    ext: str
    kind: str  # "proposal" | "script" | "reduction_json" | "iq1d" | "iqxqy2d" |
    #            "merged" | "note" | "trans" | "image" | "nexus" | "other"


@dataclass
class FolderInventory:
    """Compact, organized digest of a shared/ folder — what the LLM first sees."""

    ipts: int
    shared_dir: Path
    tree_text: str  # 2-3 level directory tree
    ext_counts: dict[str, int]
    kind_counts: dict[str, int]
    output_dirs: list[Path]  # candidate dirs holding reduced data
    proposal_pdfs: list[Path]
    scripts: list[Path]
    note_files: list[Path]
    naming_examples: list[str]  # representative output names
    total_files: int

    def as_text(self) -> str:
        def block(title: str, items: list) -> list[str]:
            out = [title]
            if items:
                out.extend(f"  - {x}" for x in items)
            else:
                out.append("  (none)")
            return out

        lines = [
            f"IPTS-{self.ipts}  shared dir: {self.shared_dir}",
            f"Total files scanned: {self.total_files}",
            "",
            "Directory tree:",
            self.tree_text,
            "",
            "File counts by kind: "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.kind_counts.items())),
            "File counts by extension: "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.ext_counts.items())),
            "",
            *block("Candidate output/data directories:", self.output_dirs),
            "",
            *block("Proposal PDFs:", self.proposal_pdfs),
            "",
            *block("Scripts found:", self.scripts),
            "",
            *block("Note/README files:", self.note_files),
            "",
            *block("Representative reduced-output names:", self.naming_examples),
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Reduction config + datasets
# --------------------------------------------------------------------------- #
@dataclass
class ReductionMeta:
    output_name: str
    ipts: Optional[int] = None
    sample_run: Optional[str] = None
    sample_thickness: Optional[float] = None
    trans_run: Optional[str] = None
    bkg_run: Optional[str] = None
    bkg_trans_run: Optional[str] = None
    empty_trans_run: Optional[str] = None
    beam_center_run: Optional[str] = None
    mask_file: Optional[str] = None
    qmin: Optional[float] = None
    qmax: Optional[float] = None
    num_q_bins: Optional[int] = None
    abs_scale: Optional[float] = None
    abs_scale_method: Optional[str] = None
    source_json: Optional[Path] = None
    config_raw: dict = field(default_factory=dict)


@dataclass
class Dataset:
    """One reduced measurement (one output_name within one variant dir)."""

    output_name: str
    variant: str  # e.g. "output" / "output_mask4"
    base: str = ""
    temperature: Optional[str] = None
    config: Optional[str] = None
    iq_path: Optional[Path] = None
    iqxqy_path: Optional[Path] = None
    merged_path: Optional[Path] = None
    trans_path: Optional[Path] = None
    meta: Optional[ReductionMeta] = None
    oncat_title: Optional[str] = None
    is_standard: bool = False

    @property
    def key(self) -> str:
        return f"{self.variant}:{self.output_name}"


# --------------------------------------------------------------------------- #
# Proposal
# --------------------------------------------------------------------------- #
@dataclass
class Hypothesis:
    text: str
    expected_signature: str = ""  # what to look for in the data


@dataclass
class ProposalInfo:
    available: bool = False
    title: Optional[str] = None
    pi: Optional[str] = None
    abstract_summary: str = ""
    science_goals: list[str] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    sample_descriptions: dict[str, str] = field(default_factory=dict)
    raw_text_chars: int = 0


# --------------------------------------------------------------------------- #
# Strategy (LLM-derived)
# --------------------------------------------------------------------------- #
@dataclass
class FitPlan:
    group_id: str
    should_fit: bool = False
    model: Optional[str] = None  # "guinier" | "porod" | "powerlaw" | None
    q_min: Optional[float] = None
    q_max: Optional[float] = None
    rationale: str = ""


@dataclass
class StrategyGroup:
    group_id: str
    label: str
    kind: str = "single"  # temperature_series|concentration_series|config_set|single
    members: list[str] = field(default_factory=list)  # output names (science only)
    comparison: str = "iq1d"  # "iq1d" | "iqxqy2d" | "both"
    ordering_key: Optional[str] = None
    description: str = ""


@dataclass
class VariantDecision:
    variants_used: list[str] = field(default_factory=list)
    compare: bool = False
    rationale: str = ""


@dataclass
class AnalysisStrategy:
    experiment_summary: str = ""
    science_goals: list[str] = field(default_factory=list)
    variant_decision: VariantDecision = field(default_factory=VariantDecision)
    groups: list[StrategyGroup] = field(default_factory=list)
    fit_plans: list[FitPlan] = field(default_factory=list)
    report_outline: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Analysis + fitting
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    kind: str
    params: dict[str, float] = field(default_factory=dict)
    q_range: tuple[float, float] = (0.0, 0.0)
    r_squared: Optional[float] = None
    ok: bool = False
    note: str = ""


@dataclass
class DatasetAnalysis:
    output_name: str
    variant: str
    q_min: float = 0.0
    q_max: float = 0.0
    n_points: int = 0
    low_q_slope: Optional[float] = None
    high_q_slope: Optional[float] = None
    fit: Optional[FitResult] = None
    flags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
@dataclass
class FigureRef:
    path: Path
    caption: str
    label: str
    width: str = "0.8\\textwidth"


@dataclass
class TableSpec:
    caption: str
    label: str
    headers: list[str]
    rows: list[list[str]]
    longtable: bool = False
    landscape: bool = False
    fontsize: str = "small"  # small | footnotesize | scriptsize | normalsize
    colspec: Optional[str] = None
    section_title: Optional[str] = None  # used when rendered as an appendix section


@dataclass
class HypothesisCheck:
    hypothesis: str
    verdict: str  # supported | not_supported | inconclusive | no_data
    evidence: str = ""
    confidence: str = "low"


@dataclass
class GroupReport:
    group: StrategyGroup
    figures: list[FigureRef] = field(default_factory=list)
    table: Optional[TableSpec] = None
    analyses: list[DatasetAnalysis] = field(default_factory=list)
    observations: str = ""


@dataclass
class ExperimentContext:
    ipts: int
    shared_dir: Path
    inventory: Optional[FolderInventory] = None
    datasets: list[Dataset] = field(default_factory=list)
    catalog: Any = None  # pandas DataFrame | None
    proposal: Optional[ProposalInfo] = None
    note_md: Optional[str] = None
    degraded: list[str] = field(default_factory=list)


@dataclass
class ReportModel:
    context: ExperimentContext
    title: str
    overview: str = ""
    catalog_table: Optional[TableSpec] = None  # main-body Sample Summary
    appendix_tables: list[TableSpec] = field(default_factory=list)
    group_reports: list[GroupReport] = field(default_factory=list)
    hypothesis_checks: list[HypothesisCheck] = field(default_factory=list)
    discussion: str = ""
    caveats: list[str] = field(default_factory=list)
    generated_at: str = ""
    model_name: str = ""
