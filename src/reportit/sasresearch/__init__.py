"""sasresearch — an autoresearch loop for SANS model fitting.

A standalone explorer that fits a single 1D I(Q) curve the way a human analyst
would: try several physically-plausible models, run each from many starting
points (to escape local minima), and rank them with a SHAPE-AWARE fitness that
judges the fit both quantitatively (reduced chi^2, log-residual size) and
qualitatively (does the model reproduce the curvature, the knee, and the
position of any bump/valley — the features a human sees first?).

Decoupled from the report pipeline so the fitting can be developed and debugged
on its own:  ``python -m reportit.sasresearch <datafile>``.
"""

__all__ = ["explore"]


def __getattr__(name):  # lazy: avoids an import cycle with analysis.sas_agent
    if name == "explore":
        from .loop import explore
        return explore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
