"""LaTeX rendering helpers for paper tables."""
from __future__ import annotations

import polars as pl


def _format_value(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        return f"{v:.4f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v).replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")


def df_to_booktabs(
    df: pl.DataFrame,
    header_map: dict[str, str],
) -> str:
    """Render a Polars DataFrame as a LaTeX booktabs `tabular` block."""
    cols = list(header_map.keys())
    headers = list(header_map.values())
    align = "l" + "r" * (len(cols) - 1)
    out = ["\\begin{tabular}{" + align + "}", "\\toprule"]
    out.append(" & ".join(headers) + " \\\\")
    out.append("\\midrule")
    if df.height == 0:
        out.append(f"\\multicolumn{{{len(cols)}}}{{c}}{{(no rows)}} \\\\")
    else:
        for row in df.select(cols).iter_rows():
            out.append(" & ".join(_format_value(v) for v in row) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out)
