"""Hand-checked metric definitions."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.metrics import compare, last_cycle_rows, rmse, score_alert_arrays, success_criteria, summarize_alerts


def test_rmse_matches_hand_calculation():
    # errors 0, 0, 3; mean square 3; rmse sqrt(3)
    score = rmse([1.0, 2.0, 3.0], [1.0, 2.0, 6.0])
    assert math.isclose(score, math.sqrt(3.0), rel_tol=0, abs_tol=1e-12)


def test_compare_boundaries():
    assert compare(18.0, "<=", 18.0)
    assert not compare(18.000001, "<=", 18.0)
    assert compare(0.8, ">=", 0.8)
    assert compare(0.1, "<=", 0.1)
    assert not compare(0.100001, "<=", 0.1)
    assert compare(5, "between", 1, 5)
    assert compare(None, "<=", 1) is False


def test_success_criteria_names_and_failure():
    rows = success_criteria(19.0, 10.0, 0.9, 0.05, 80.0)
    names = [row["name"] for row in rows]
    assert names == [
        "FD001 test RUL RMSE",
        "FD003 test RUL RMSE",
        "Alert lead time",
        "False alerts",
        "Test coverage of src/",
    ]
    assert rows[0]["passed"] is False
    assert rows[1]["passed"] is True
    assert rows[2]["passed"] is True
    assert rows[3]["passed"] is True
    assert rows[4]["passed"] is True


def test_alert_scores_match_a_worked_example():
    # Engine A: first alert at RUL 30, a later episode at RUL 0. Neither starts above 50.
    # Engine B: one episode starting at RUL 80, two cycles of that episode are still above 50.
    alerts = [
        np.array([False, True, True, False, True]),
        np.array([True, True, False]),
    ]
    ruls = [
        np.array([40.0, 30.0, 20.0, 10.0, 0.0]),
        np.array([80.0, 70.0, 60.0]),
    ]
    stats = score_alert_arrays(alerts, ruls, lead_min=15, false_rul=50)
    assert stats["n_engines"] == 2
    assert stats["lead_time_fraction"] == 1.0
    assert stats["false_alerts_per_engine"] == 0.5
    assert stats["false_alert_cycles_per_engine"] == 1.0
    assert stats["median_lead_time"] == 55.0


def test_no_alert_has_zero_lead_and_is_not_a_false_alert():
    stats = score_alert_arrays([np.array([False, False])], [np.array([80.0, 10.0])])
    assert stats["lead_time_fraction"] == 0.0
    assert stats["median_lead_time"] == 0.0
    assert stats["false_alerts_per_engine"] == 0.0


def test_summarize_agrees_with_array_scores():
    cycles = pd.DataFrame(
        {
            "dataset": ["FD001"] * 5 + ["FD003"] * 3,
            "unit_id": [1, 1, 1, 1, 1, 7, 7, 7],
            "cycle": [1, 2, 3, 4, 5, 1, 2, 3],
            "rul_true": [40, 30, 20, 10, 0, 80, 70, 60],
            "alert": [0, 1, 1, 0, 1, 1, 1, 0],
        }
    )
    engines = summarize_alerts(cycles)
    assert set(engines["unit_id"]) == {1, 7}
    row_a = engines[engines["unit_id"] == 1].iloc[0]
    row_b = engines[engines["unit_id"] == 7].iloc[0]
    assert row_a["lead_time"] == 30
    assert row_a["false_alert_episodes"] == 0
    assert int(row_a["meets_lead_time_15"]) == 1
    assert row_b["lead_time"] == 80
    assert row_b["false_alert_episodes"] == 1
    assert row_b["false_alert_cycles"] == 2


def test_last_cycle_keeps_the_final_row():
    frame = pd.DataFrame({"unit_id": [1, 1, 2], "cycle": [1, 5, 3], "value": [10, 11, 7]})
    last = last_cycle_rows(frame)
    assert set(zip(last["unit_id"], last["value"])) == {(1, 11), (2, 7)}
