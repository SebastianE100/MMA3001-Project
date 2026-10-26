"""pytest tests for src/occupancy.py. Run from the repository root with: pytest"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.polynomial import Polynomial
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import occupancy as occ  # noqa: E402


# ---------- fixtures ----------

@pytest.fixture
def raw():
    """Three days of readings (UTC): 1 Jan has data, 2 Jan has none, 3 Jan has data."""
    return pd.DataFrame({
        "collecteddate": [
            "2024-03-04T00:10:00Z", "2024-03-04T00:40:00Z",   # same UTC hour
            "2024-03-04T03:00:00Z",
            "2024-03-06T01:00:00Z",                           # 2 days later
        ],
        "headcount": [3, 7, 5, 4],
    })


@pytest.fixture
def fitted_tree():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "dayNum": rng.integers(0, 7, 300),
        "Time": rng.integers(0, 24, 300).astype(float),
        "Class": rng.integers(0, 2, 300).astype(bool),
        "Exam": rng.integers(0, 2, 300).astype(bool),
    })
    y = X["Time"].between(9, 17) * 10.0 + rng.normal(0, 1, 300)
    return DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)


# ---------- prepare_data ----------

def test_hourly_maximum_is_used(raw):
    out = occ.prepare_data(raw)
    # 00:10 and 00:40 UTC = 11:10 and 11:40 Melbourne (AEDT), max of (3, 7) is 7
    row = out[(out["Date"] == pd.Timestamp("2024-03-04").date()) & (out["Time"] == 11)]
    assert row["headcount"].iloc[0] == 7


def test_timestamps_are_melbourne_time(raw):
    out = occ.prepare_data(raw)
    assert str(out["collecteddate"].dt.tz) == "Australia/Melbourne"


def test_empty_hours_are_zero_filled(raw):
    out = occ.prepare_data(raw)
    assert out["headcount"].notna().all()
    assert (out["headcount"] == 0).any()


def test_day_with_no_readings_is_removed(raw):
    out = occ.prepare_data(raw)
    assert pd.Timestamp("2024-03-05").date() not in set(out["Date"])
    assert pd.Timestamp("2024-03-04").date() in set(out["Date"])
    assert pd.Timestamp("2024-03-06").date() in set(out["Date"])


def test_duplicate_timestamps_keep_first():
    dup = pd.DataFrame({
        "collecteddate": ["2024-03-04T00:10:00Z", "2024-03-04T00:10:00Z"],
        "headcount": [4, 99],
    })
    out = occ.prepare_data(dup)
    assert out["headcount"].max() == 4


def test_calendar_columns(raw):
    out = occ.prepare_data(raw)
    assert set(out["Day"]) <= set(occ.DAY_MAP)
    assert out["Time"].between(0, 23.75).all()
    monday = out[out["Date"] == pd.Timestamp("2024-03-04").date()]
    assert (monday["Day"] == "Monday").all()


def test_class_and_exam_flags(raw):
    out = occ.prepare_data(
        raw,
        teaching_periods=[("2024-03-04", "2024-03-05")],
        exam_periods=[("2024-03-06", "2024-03-07")],
    )
    d4 = out[out["Date"] == pd.Timestamp("2024-03-04").date()]
    d6 = out[out["Date"] == pd.Timestamp("2024-03-06").date()]
    assert d4["Class"].any() and not d4["Exam"].any()
    assert d6["Exam"].any() and not d6["Class"].any()


def test_flag_boundaries_are_inclusive():
    ts = pd.Series(pd.to_datetime(["2024-05-27 00:00"]).tz_localize(occ.TZ))
    assert occ.in_periods(ts, [("2024-02-26", "2024-05-27")]).iloc[0]
    assert occ.in_periods(ts, [("2024-05-27", "2024-06-21")]).iloc[0]


def test_missing_column_raises(raw):
    with pytest.raises(ValueError, match="Missing required columns"):
        occ.prepare_data(raw.drop(columns=["headcount"]))


def test_negative_headcount_raises(raw):
    raw.loc[0, "headcount"] = -1
    with pytest.raises(ValueError, match="negative"):
        occ.prepare_data(raw)


def test_unparseable_timestamp_raises(raw):
    raw.loc[0, "collecteddate"] = "not a date"
    with pytest.raises(ValueError, match="unparseable"):
        occ.prepare_data(raw)


# ---------- chronological validation ----------

def test_train_always_precedes_validation():
    X = pd.DataFrame({"a": range(500)})
    for train, val in TimeSeriesSplit(n_splits=5).split(X):
        assert train.max() < val.min()


def test_validation_blocks_do_not_overlap():
    X = pd.DataFrame({"a": range(500)})
    seen = set()
    for _, val in TimeSeriesSplit(n_splits=5).split(X):
        assert seen.isdisjoint(val)
        seen.update(val)


# ---------- baseline ----------

def test_mean_baseline_values():
    times = pd.Series([9.0, 9.0, 10.0])
    y = pd.Series([2.0, 4.0, 10.0])
    base = occ.fit_mean_baseline(times, y)
    assert base == {"9.0": 3.0, "10.0": 10.0}


def test_mean_baseline_unseen_time_is_nan():
    base = occ.fit_mean_baseline(pd.Series([9.0]), pd.Series([5.0]))
    pred = occ.predict_mean_baseline(base, pd.Series([9.0, 12.0]))
    assert pred.iloc[0] == 5.0 and np.isnan(pred.iloc[1])


def test_baseline_is_perfect_on_constant_pattern():
    times = pd.Series([0.0, 1.0] * 10)
    y = pd.Series([1.0, 5.0] * 10)
    pred = occ.predict_mean_baseline(occ.fit_mean_baseline(times, y), times)
    assert r2_score(y, pred) == pytest.approx(1.0)


# ---------- polynomial model ----------

def test_converted_coefficients_match_fitted_polynomial():
    """Guards against the earlier bug: raw .coef values need .convert() and
    lowest-power-first evaluation."""
    x = np.linspace(0, 23, 24)
    y = 0.05 * x**2 - 1.2 * x + 3.0
    p = Polynomial.fit(x, y, deg=2)
    coef = p.convert().coef
    via_coef = np.polynomial.polynomial.polyval(x, coef)
    assert np.allclose(via_coef, p(x))
    assert np.allclose(via_coef, y)


# ---------- weights and error rates ----------

def test_zero_weights():
    w = occ.zero_weights([0, 3, 0, 8], 5)
    assert list(w) == [5, 1, 5, 1]


def test_empty_rates_known_case():
    y_true = [0, 1, 2, 5, 9, 10]      # empty: 0,1,2  occupied: 5,9,10
    y_pred = [0, 4, 1, 1, 8, 9]       # empty hour 1 predicted 4 (FP), occupied 5 predicted 1 (FN)
    fp, fn = occ.empty_rates(y_true, y_pred)
    assert fp == pytest.approx(1 / 3)
    assert fn == pytest.approx(1 / 3)


def test_empty_threshold_is_inclusive():
    fp, _ = occ.empty_rates([2], [2])      # 2 people counts as empty, predicted empty
    assert fp == 0.0


def test_empty_rates_nan_when_class_absent():
    fp, fn = occ.empty_rates([5, 6], [5, 6])
    assert np.isnan(fp) and fn == 0.0


# ---------- predict_headcount ----------

def test_prediction_is_nonnegative_and_flags_occupied(fitted_tree):
    out = occ.predict_headcount(fitted_tree, [0, 1], [3.0, 13.0], [True, True], [False, False])
    assert (out["headcount_pred"] >= 0).all()
    assert list(out["occupied"]) == [False, True]


def test_scalar_and_array_inputs_agree(fitted_tree):
    a = occ.predict_headcount(fitted_tree, 2, 10.0, True, False)
    b = occ.predict_headcount(fitted_tree, [2], [10.0], [True], [False])
    assert a.equals(b)


def test_output_length_matches_input(fitted_tree):
    out = occ.predict_headcount(fitted_tree, range(7), [12.0] * 7, [True] * 7, [False] * 7)
    assert len(out) == 7


@pytest.mark.parametrize("day", [-1, 7, 2.5])
def test_bad_day_rejected(fitted_tree, day):
    with pytest.raises(ValueError, match="dayNum"):
        occ.predict_headcount(fitted_tree, day, 10.0, True, False)


@pytest.mark.parametrize("time", [-0.5, 24.0, 30.0])
def test_bad_time_rejected(fitted_tree, time):
    with pytest.raises(ValueError, match="time"):
        occ.predict_headcount(fitted_tree, 1, time, True, False)


def test_missing_value_rejected(fitted_tree):
    with pytest.raises(ValueError, match="missing"):
        occ.predict_headcount(fitted_tree, 1, np.nan, True, False)


def test_mismatched_lengths_rejected(fitted_tree):
    with pytest.raises(ValueError):
        occ.predict_headcount(fitted_tree, [1, 2], [10.0], [True], [False])


def test_period_end_date_means_midnight_at_start_of_that_day():
    """Matches DataPrep.py: an end date such as 2024-05-27 is 00:00 on that
    day, so the daytime hours of the end date are NOT inside the period."""
    ts = pd.Series(pd.to_datetime(["2024-05-27 12:00"]).tz_localize(occ.TZ))
    assert not occ.in_periods(ts, [("2024-02-26", "2024-05-27")]).iloc[0]
