"""JSON-schema specs for the read-only probe tools + finalize_strategy.

These are exposed to the strategist LLM via chat_with_tools(). The strategist
calls probes to inspect the folder, then calls finalize_strategy once it knows
what the experiment is and how to report it.
"""

from __future__ import annotations

FINALIZE_TOOL = "finalize_strategy"


def _fn(name, description, properties, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


PROBE_TOOLS = [
    _fn("list_dir",
        "List the entries (files + subdirs, with sizes) of a directory inside the "
        "experiment's shared folder. Use to explore subfolders.",
        {"path": {"type": "string", "description": "Path relative to the shared dir, or absolute within it."}},
        ["path"]),
    _fn("read_text",
        "Read up to max_bytes of a text file (NOTE.md, README, a .py reduction "
        "script, a .txt). Use to understand what was done.",
        {"path": {"type": "string"},
         "max_bytes": {"type": "integer", "description": "default 6000"}},
        ["path"]),
    _fn("head_file",
        "Return the first N lines of a (possibly large) data file — e.g. to see "
        "the header/columns of an _Iq.dat or _Iqxqy.dat.",
        {"path": {"type": "string"}, "n": {"type": "integer", "description": "default 15"}},
        ["path"]),
    _fn("parse_reduction_json",
        "Parse a per-sample reduction config .json into structured fields "
        "(sample/background/transmission run numbers, thickness, mask, Qmin/Qmax, "
        "absolute scale).",
        {"path": {"type": "string"}},
        ["path"]),
    _fn("oncat_titles",
        "Look up the ONCat run TITLE (and key metadata) for one or more run "
        "numbers. Titles often reveal what a sample is.",
        {"runs": {"type": "array", "items": {"type": "string"},
                  "description": "Run numbers as strings."}},
        ["runs"]),
    _fn("sample_curve",
        "Get a downsampled (Q, I) representation of a 1D dataset so you can 'see' "
        "its shape (flat, power-law, peak, Guinier knee). Provide the output name "
        "and variant (output dir name).",
        {"output_name": {"type": "string"}, "variant": {"type": "string"},
         "points": {"type": "integer", "description": "default 25"}},
        ["output_name"]),
    _fn("list_datasets",
        "List all discovered reduced datasets (output names) with their parsed "
        "base/temperature/config/variant and whether they are calibration "
        "standards. Use to plan grouping.",
        {"variant": {"type": "string", "description": "Optional: restrict to one variant/output dir."}},
        []),
]


FINALIZE_STRATEGY = _fn(
    FINALIZE_TOOL,
    "Record the final analysis strategy for the report. Call this exactly once "
    "when you understand the experiment and how it should be reported.",
    {
        "experiment_summary": {"type": "string",
            "description": "2-5 sentences: what this experiment measured and why."},
        "science_goals": {"type": "array", "items": {"type": "string"}},
        "variant_decision": {
            "type": "object",
            "properties": {
                "variants_used": {"type": "array", "items": {"type": "string"},
                    "description": "Which output dir(s) to use, e.g. ['output_mask4'] or ['output','output_mask4']."},
                "compare": {"type": "boolean",
                    "description": "If true, overlay variants to compare them."},
                "rationale": {"type": "string"},
            },
            "required": ["variants_used", "rationale"],
        },
        "curve_source": {
            "type": "string",
            "enum": ["combined", "individual", "auto"],
            "description": "Which 1D curves to plot. 'combined' = use the merged/"
            "stitched extended-Q profiles (check the inventory's combined-files "
            "list — they may be named merged_*, *_stitched, etc.); 'individual' = "
            "use per-configuration *_Iq.dat curves (choose this if NO combined "
            "files exist, or to show each configuration separately); 'auto' = "
            "combined where available else individual. Decide from what actually "
            "exists in the folder.",
        },
        "curve_source_rationale": {"type": "string"},
        "groups": {
            "type": "array",
            "description": "How to group datasets into comparison figures. Exclude calibration standards (porsil).",
            "items": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "label": {"type": "string"},
                    "kind": {"type": "string",
                        "enum": ["temperature_series", "concentration_series", "config_set", "single", "other"]},
                    "members": {"type": "array", "items": {"type": "string"},
                        "description": "output names belonging to this group."},
                    "comparison": {"type": "string", "enum": ["iq1d", "iqxqy2d", "both"]},
                    "ordering_key": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["group_id", "label", "members", "comparison"],
            },
        },
        "fit_plans": {
            "type": "array",
            "description": "Per-group decision on whether a model fit is sensible.",
            "items": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string"},
                    "should_fit": {"type": "boolean"},
                    "model": {"type": "string",
                        "enum": ["guinier", "correlation", "porod", "powerlaw", "none"],
                        "description": "guinier = compact-particle Rg (ONLY if a clear "
                        "low-Q plateau/knee exists); correlation = Ornstein-Zernike "
                        "correlation length xi (solution scattering with a low-Q plateau "
                        "rolling into a power law); porod/powerlaw = interfacial/network "
                        "slope. Choose based on the actual curve shape from sample_curve."},
                    "q_min": {"type": "number"},
                    "q_max": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["group_id", "should_fit"],
            },
        },
        "report_outline": {"type": "array", "items": {"type": "string"},
            "description": "Ordered list of report section titles."},
        "caveats": {"type": "array", "items": {"type": "string"}},
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    ["experiment_summary", "groups"],
)


def all_tools() -> list[dict]:
    return PROBE_TOOLS + [FINALIZE_STRATEGY]
