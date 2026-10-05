"""Static HTML operations dashboard. Plotly JS is embedded so the file opens offline."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def _fig_html(fig: go.Figure, include_js: bool) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=include_js)


def write_dashboard(
    path: Path,
    metrics: dict,
    fd001: pd.DataFrame,
    fd003: pd.DataFrame,
    cycles: pd.DataFrame,
    engines: pd.DataFrame,
) -> None:
    rows = []
    for item in metrics["success_criteria"]:
        value = "n/a" if item["value"] is None else item["value"]
        status = "pass" if item["passed"] else "fail"
        rows.append(
            "<tr>"
            f"<td>{escape(str(item['name']))}</td>"
            f"<td>{escape(str(value))}</td>"
            f"<td>{escape(str(item['op']))} {escape(str(item['threshold']))}</td>"
            f"<td>{status}</td>"
            "</tr>"
        )
    table = (
        "<table><thead><tr><th>Criterion</th><th>Value</th><th>Threshold</th><th>Result</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    figures: list[str] = []
    include_js = True
    if len(fd001) and len(fd003):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fd001["rul_true"], y=fd001["rul_pred"], mode="markers", name="FD001"))
        fig.add_trace(go.Scatter(x=fd003["rul_true"], y=fd003["rul_pred"], mode="markers", name="FD003"))
        fig.update_layout(
            title="Official test: predicted vs true RUL",
            xaxis_title="True RUL (cycles)",
            yaxis_title="Predicted RUL (cycles)",
        )
        figures.append(_fig_html(fig, include_js))
        include_js = False
    if len(engines):
        fig = go.Figure()
        for dataset in ("FD001", "FD003"):
            subset = engines[engines["dataset"] == dataset]
            fig.add_trace(go.Bar(x=subset["unit_id"].astype(str), y=subset["lead_time"], name=dataset))
        fig.update_layout(title="Validation alert lead time by engine", barmode="group", yaxis_title="Cycles")
        figures.append(_fig_html(fig, include_js))
        include_js = False
    if len(cycles):
        pool = engines[engines["dataset"] == "FD001"] if len(engines) else engines
        if len(pool):
            median = float(pool["lead_time"].median())
            pick = pool.iloc[(pool["lead_time"] - median).abs().argmin()]
            subset = cycles[(cycles["dataset"] == pick["dataset"]) & (cycles["unit_id"] == pick["unit_id"])]
            subset = subset.sort_values("cycle")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=subset["cycle"], y=subset["rul_true"], name="True RUL", mode="lines"))
            fig.add_trace(go.Scatter(x=subset["cycle"], y=subset["rul_pred"], name="Predicted RUL", mode="lines"))
            alerts = subset[subset["alert"].astype(bool)]
            fig.add_trace(go.Scatter(x=alerts["cycle"], y=alerts["rul_pred"], name="Alert", mode="markers"))
            fig.update_layout(title=f"Alert review: {pick['dataset']} unit {int(pick['unit_id'])}")
            figures.append(_fig_html(fig, include_js))
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Predictive maintenance dashboard</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem; color: #1c1c1c; }}
table {{ border-collapse: collapse; margin-bottom: 1.5rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.7rem; text-align: left; }}
th {{ background: #f4f1ea; }}
</style>
</head>
<body>
<h1>Turbofan predictive maintenance</h1>
<p>Validation alerts and official test RUL scores. Numbers come from the prediction tables, not from a hand-edited summary.</p>
{table}
{''.join(figures)}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
