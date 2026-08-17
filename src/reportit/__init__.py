"""reportit — LLM-driven post-experiment report generator for EQSANS."""

# SINGLE SOURCE OF TRUTH for the version. pyproject.toml reads it from here
# (hatch dynamic version), so the two can never drift apart.
#
# Versioning policy (see CHANGELOG.md):
#   +0.0.1  small change — bug fix, wording, tuning, config/model swap
#   +0.1.0  major function addition — a new capability or CLI flag
__version__ = "0.11.0"
