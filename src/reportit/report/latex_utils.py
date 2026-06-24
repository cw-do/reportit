"""LaTeX escaping, Unicode sanitization, and small formatting helpers.

LLM- and metric-generated text frequently contains Unicode (Å, ∈, ≈, °, ², ⁻¹,
Greek letters). pdflatex's inputenc rejects unmapped Unicode, so we translate a
known set to LaTeX and drop anything else non-ASCII.
"""

from __future__ import annotations

_ESCAPE = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# Unicode -> LaTeX (text-mode safe; \ensuremath works in and out of math).
_UNICODE = {
    "Å": r"\AA{}", "Â": "A", "µ": r"\ensuremath{\mu}", "μ": r"\ensuremath{\mu}",
    "∈": r"\ensuremath{\in}", "≈": r"\ensuremath{\approx}",
    "≤": r"\ensuremath{\leq}", "≥": r"\ensuremath{\geq}",
    "×": r"\ensuremath{\times}", "·": r"\ensuremath{\cdot}",
    "±": r"\ensuremath{\pm}", "→": r"\ensuremath{\rightarrow}",
    "←": r"\ensuremath{\leftarrow}", "↑": r"\ensuremath{\uparrow}",
    "↓": r"\ensuremath{\downarrow}", "∝": r"\ensuremath{\propto}",
    "°": r"\ensuremath{^\circ}", "∞": r"\ensuremath{\infty}",
    "²": r"\ensuremath{^2}", "³": r"\ensuremath{^3}",
    "⁰": r"\ensuremath{^0}", "¹": r"\ensuremath{^1}",
    "⁴": r"\ensuremath{^4}", "⁻": r"\ensuremath{^-}",
    "½": r"\ensuremath{\tfrac12}", "√": r"\ensuremath{\sqrt{}}",
    "α": r"\ensuremath{\alpha}", "β": r"\ensuremath{\beta}",
    "γ": r"\ensuremath{\gamma}", "δ": r"\ensuremath{\delta}",
    "θ": r"\ensuremath{\theta}", "λ": r"\ensuremath{\lambda}",
    "π": r"\ensuremath{\pi}", "ρ": r"\ensuremath{\rho}",
    "σ": r"\ensuremath{\sigma}", "φ": r"\ensuremath{\phi}",
    "χ": r"\ensuremath{\chi}", "ω": r"\ensuremath{\omega}",
    "Δ": r"\ensuremath{\Delta}", "Σ": r"\ensuremath{\Sigma}",
    "–": "--", "—": "---", "‐": "-", "−": r"\ensuremath{-}",
    "’": "'", "‘": "'", "“": "``", "”": "''", "…": r"\ldots{}",
    " ": " ", "\t": " ",
}


def escape(text) -> str:
    """Escape LaTeX specials and translate Unicode; drop unknown non-ASCII."""
    if text is None:
        return ""
    out = []
    for ch in str(text):
        if ch in _ESCAPE:
            out.append(_ESCAPE[ch])
        elif ch in _UNICODE:
            out.append(_UNICODE[ch])
        elif ch == "\n" or (32 <= ord(ch) < 127):
            out.append(ch)
        else:
            out.append("")  # drop other non-ASCII
    return "".join(out)


def _sanitize_math(seg: str) -> str:
    """Inside $...$: keep ASCII LaTeX, translate known Unicode, drop the rest."""
    out = []
    for ch in seg:
        if ch in _UNICODE:
            # strip an outer \ensuremath{...} since we're already in math
            v = _UNICODE[ch]
            if v.startswith(r"\ensuremath{") and v.endswith("}"):
                v = v[len(r"\ensuremath{"):-1]
            out.append(v)
        elif 32 <= ord(ch) < 127:
            out.append(ch)
        else:
            out.append("")
    return "".join(out)


def escape_keep_math(text) -> str:
    """Escape text but leave $...$ math spans as LaTeX (sanitizing Unicode there)."""
    if text is None:
        return ""
    parts = str(text).split("$")
    for k in range(len(parts)):
        parts[k] = escape(parts[k]) if k % 2 == 0 else _sanitize_math(parts[k])
    return "$".join(parts)
