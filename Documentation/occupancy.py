"""Reusable functions for the building occupancy forecasting project.

The project scripts (DataPrep.py, Baseline_Model.py, Weight_Sweep.py, ...) run
top to bottom with hard-coded paths, so they cannot be imported and tested.
The logic below is the same logic, wrapped in functions so that it can be
tested with pytest and imported by the scripts.
"""

import numpy as np
import pandas as pd

TZ = "Australia/Melbourne"
EMPTY_THRESHOLD = 2  # people; empty means headcount <= this value
DAY_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}


def prepare_data(raw, teaching_periods=(), exam_periods=()):
    """Clean raw sensor readings and build the modelling table.

    Mirrors DataPrep.py: remove duplicate timestamps, convert to Melbourne
    time, resample to hourly maxima, set empty hours to 0, drop calendar days
    that had no readings at all, and add calendar columns.

    Args:
        raw (pandas.DataFrame): Columns ``collecteddate`` (ISO 8601 strings or
            datetimes, read as UTC if no timezone) and ``headcount`` (number).
        teaching_periods (iterable of (str, str)): (start, end) dates of
            teaching periods, as "YYYY-MM-DD" strings in Melbourne time.
        exam_periods (iterable of (str, str)): Same, for exam periods.

    Returns:
        pandas.DataFrame: Columns ``collecteddate`` (Melbourne time, hourly),
        ``headcount``, ``Day``, ``Date``, ``Time`` (hour of day as float),
        ``Class`` and ``Exam`` (bool).

    Raises:
        ValueError: If a required column is missing, a timestamp cannot be
            parsed, or any headcount is negative.
    """
    missing = {"collecteddate", "headcount"} - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = raw[["collecteddate", "headcount"]].copy()
    df = df.drop_duplicates(subset=["collecteddate"])
    try:
        df["collecteddate"] = pd.to_datetime(
            df["collecteddate"], format="ISO8601", utc=True
        ).dt.tz_convert(TZ)
    except (ValueError, TypeError) as err:
        raise ValueError("collecteddate contains unparseable timestamps") from err
    if (df["headcount"] < 0).any():
        raise ValueError("headcount must not be negative")

    days_with_data = set(df["collecteddate"].dt.date)

    df = (
        df.resample("60min", on="collecteddate")["headcount"]
        .max()
        .reset_index()
        .fillna({"headcount": 0})
    )
    df["Date"] = df["collecteddate"].dt.date
    df = df[df["Date"].isin(days_with_data)].copy()
    df["Day"] = df["collecteddate"].dt.day_name()
    df["Time"] = df["collecteddate"].dt.hour + df["collecteddate"].dt.minute / 60
    df["Class"] = in_periods(df["collecteddate"], teaching_periods)
    df["Exam"] = in_periods(df["collecteddate"], exam_periods)
    return df.reset_index(drop=True)


def in_periods(timestamps, periods):
    """Return a boolean Series, True where a timestamp is inside any period.

    Args:
        timestamps (pandas.Series): Timezone-aware (Melbourne) datetimes.
        periods (iterable of (str, str)): (start, end) date strings. Both ends
            are inclusive, as in DataPrep.py.

    Returns:
        pandas.Series: Boolean, same index as ``timestamps``.
    """
    flag = pd.Series(False, index=timestamps.index)
    for start, end in periods:
        lo = pd.Timestamp(start).tz_localize(TZ)
        hi = pd.Timestamp(end).tz_localize(TZ)
        flag |= timestamps.between(lo, hi)
    return flag


def fit_mean_baseline(times, headcount):
    """Fit the Mean Time baseline: average headcount for each time of day.

    Args:
        times (pandas.Series): Time of day in hours (training rows only).
        headcount (pandas.Series): Headcount for the same rows.

    Returns:
        dict: Maps ``str(time)`` to the mean headcount at that time.
    """
    return {str(t): float(np.mean(headcount[times == t])) for t in times.unique()}


def predict_mean_baseline(baseline, times):
    """Predict with a Mean Time baseline. Unseen times give NaN.

    Args:
        baseline (dict): Output of :func:`fit_mean_baseline`.
        times (pandas.Series): Time of day in hours.

    Returns:
        pandas.Series: Predicted headcount, NaN for times not in training.
    """
    return times.astype(str).map(baseline)


def zero_weights(headcount, weight):
    """Sample weights that up-weight empty (zero-headcount) hours.

    Args:
        headcount (array-like): Training headcounts.
        weight (float): Weight given to rows with headcount equal to 0.

    Returns:
        numpy.ndarray: ``weight`` where headcount is 0, otherwise 1.
    """
    return np.where(np.asarray(headcount) == 0, weight, 1)


def empty_rates(y_true, y_pred, threshold=EMPTY_THRESHOLD):
    """False positive and false negative rates for the empty/occupied call.

    Empty means headcount <= ``threshold``. A false positive is an empty hour
    predicted occupied (wasted energy). A false negative is an occupied hour
    predicted empty (discomfort).

    Args:
        y_true (array-like): Actual headcounts.
        y_pred (array-like): Predicted headcounts.
        threshold (float): Empty/occupied boundary.

    Returns:
        tuple of float: (false_positive_rate, false_negative_rate). A rate is
        NaN if its class has no members.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    empty = y_true <= threshold
    occupied = ~empty
    fp = ((y_pred > threshold) & empty).sum() / empty.sum() if empty.any() else np.nan
    fn = ((y_pred <= threshold) & occupied).sum() / occupied.sum() if occupied.any() else np.nan
    return fp, fn


def predict_headcount(tree, day_num, time, class_flag, exam_flag):
    """Predict hourly building headcount from calendar inputs.

    Args:
        tree (sklearn.tree.DecisionTreeRegressor): Tree fitted on the features
            ["dayNum", "Time", "Class", "Exam"], in that order.
        day_num (int or array-like): 0 (Monday) to 6 (Sunday).
        time (float or array-like): Hours, from 0 up to but not including 24.
        class_flag (bool or array-like): True during a teaching period.
        exam_flag (bool or array-like): True during an exam period.

    Returns:
        pandas.DataFrame: Columns ``headcount_pred`` (float, 0 or more) and
        ``occupied`` (bool, True if prediction is above EMPTY_THRESHOLD).

    Raises:
        ValueError: If inputs differ in length, contain missing values, if
            day_num is not an integer from 0 to 6, or time is outside [0, 24).
    """
    inputs = pd.DataFrame({
        "dayNum": np.atleast_1d(day_num),
        "Time": np.atleast_1d(time),
        "Class": np.atleast_1d(class_flag),
        "Exam": np.atleast_1d(exam_flag),
    })
    if inputs.isna().any().any():
        raise ValueError("Inputs must not contain missing values.")
    if not inputs["dayNum"].between(0, 6).all() or (inputs["dayNum"] % 1 != 0).any():
        raise ValueError("dayNum must be an integer from 0 (Monday) to 6 (Sunday).")
    if not ((inputs["Time"] >= 0) & (inputs["Time"] < 24)).all():
        raise ValueError("time must be in hours from 0 up to, but not including, 24.")

    inputs = inputs.astype({"dayNum": int, "Time": float, "Class": bool, "Exam": bool})
    pred = np.clip(tree.predict(inputs[["dayNum", "Time", "Class", "Exam"]]), 0, None)
    return pd.DataFrame({"headcount_pred": pred, "occupied": pred > EMPTY_THRESHOLD})
