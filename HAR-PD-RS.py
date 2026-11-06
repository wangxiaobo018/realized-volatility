# -*- coding: utf-8 -*-
"""
HAR-PD-RS

Aligned with the supplied HAR-PD-RV architecture.

Core rules
----------
- 5-minute file : data_idx_399006_merged.csv
- daily file    : day_data_399006_merged.csv
- code          : 399006.XSHE
- start         : 2015-01-01
- H1/H5/H22 use the SAME 1000 common OOS origins
- target at origin t = mean RV_(t+1),...,RV_(t+h)
- PD kernels at t use t-1 and earlier information only
- fixed-length rolling raw estimation window
- chronological last 5% validation MSE
- h-origin purge between validation fit and validation blocks
- scaled Powell objective; raw MSE ranks candidates
- scalable 15-point diagonal/coordinate grid audit + local Powell
- lambda bounds = [0.0001, 5.0]
- lambda >= 4.99 is a boundary estimate
- boundary treatment = 0.1% profile-equivalent interior replacement
- if no equivalent interior solution exists, keep the boundary optimum
- negative/nonfinite RV forecasts are replaced by 1e-5
- final regression = OLS
"""

import os
import warnings
from collections import OrderedDict
from time import perf_counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.optimize import minimize
from scipy.signal import lfilter
from statsmodels.tools.tools import pinv_extended
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



warnings.filterwarnings("ignore")


# ============================================================
# 0. Settings
# ============================================================

INTRADAY_FILE = "c:/Users/lenovo/Desktop/HAR/data_idx_399006_merged.csv"
DAILY_FILE = "c:/Users/lenovo/Desktop/HAR/day_data_399006_merged.csv"

CODE = "399006.XSHE"
START_DATE = "2015-01-01"

FORECAST_WINDOW = 1000
COMMON_MAX_HORIZON = 22
HORIZONS = [1, 5, 22]

# Match the actual value in the supplied HAR-PD-RV code.
VALIDATION_FRACTION = 0.05

LAMBDA_LOWER = 0.0001
LAMBDA_UPPER = 5.0

BOUNDARY_TOL = 0.01
UPPER_BOUNDARY_THRESHOLD = LAMBDA_UPPER - BOUNDARY_TOL
INTERIOR_UPPER = UPPER_BOUNDARY_THRESHOLD - 1e-6

# 0.1% profile-equivalence rule
PROFILE_REL_TOL = 0.001
PROFILE_ABS_TOL = 1e-14
PROFILE_GRID_SIZE = 25
PROFILE_LOCAL_STARTS = 5

GRID_SIZE = 15
GRID_LOCAL_STARTS = 3
OBJECTIVE_SCALE_FLOOR = 1e-24

MODEL_LABEL = "HAR-PD-RS"
FILE_PREFIX = "HAR-PD-RS"

SOURCE_COLS = ['returns', 'RS_plus', 'RS_minus']
PD_NAMES = ['R1T', 'PDRS_plus', 'PDRS_minus']
LAMBDA_NAMES = ['lambda_R1T', 'lambda_RS_plus', 'lambda_RS_minus']
COMPONENT_COLS = ['RS_plus', 'RS_minus']

FEATURE_COLS = [
    f"{name}_lag{lag}"
    for name in PD_NAMES
    for lag in (1, 5, 22)
]

_FAST_CONTEXT = None




# ============================================================
# 1. Daily realized-volatility decomposition
# ============================================================


def calculate_daily_components(group):
    group = group.sort_values("datetime").copy()
    price = group["close"].to_numpy(dtype=float)

    if (
        len(price) < 3
        or np.any(~np.isfinite(price))
        or np.any(price <= 0)
    ):
        return pd.Series({
            "RV": np.nan,
            "RS_plus": np.nan,
            "RS_minus": np.nan,
        })

    ret = np.diff(np.log(price))

    if len(ret) < 2:
        return pd.Series({
            "RV": np.nan,
            "RS_plus": np.nan,
            "RS_minus": np.nan,
        })

    RV = float(np.sum(ret ** 2))
    RS_plus = float(np.sum(ret[ret > 0] ** 2))
    RS_minus = float(np.sum(ret[ret < 0] ** 2))

    return pd.Series({
        "RV": RV,
        "RS_plus": RS_plus,
        "RS_minus": RS_minus,
    })



# ============================================================
# 2. Causal exponential path-dependent kernel
# ============================================================

def compute_kernel_recursive(x, lam):
    x = np.asarray(x, dtype=float)
    n = len(x)

    out = np.full(n, np.nan)

    if n <= 1:
        return out

    decay = np.exp(-lam)

    raw_val = lfilter(
        [lam],
        [1.0, -decay],
        np.concatenate([[0.0], x[:-1]])
    )

    raw_w = lfilter(
        [lam],
        [1.0, -decay],
        np.concatenate([[0.0], np.ones(n - 1)])
    )

    valid = raw_w > 1e-15

    out[1:] = np.where(
        valid[1:],
        raw_val[1:] / raw_w[1:],
        np.nan
    )

    return out


# ============================================================
# 3. Future-average RV target
# ============================================================

def future_average_rv(rv_values, horizon):
    rv_values = np.asarray(rv_values, dtype=float)
    n = len(rv_values)

    out = np.full(n, np.nan)

    for i in range(n):
        if i + horizon < n:
            out[i] = np.mean(
                rv_values[i + 1:i + horizon + 1]
            )

    return out


# ============================================================
# 4. PD feature construction
# ============================================================

def add_pd_features(frame, lambda_vec):
    lambda_vec = np.asarray(lambda_vec, dtype=float)

    if lambda_vec.shape != (len(LAMBDA_NAMES),):
        raise ValueError(
            f"Expected {len(LAMBDA_NAMES)} lambdas, "
            f"got {lambda_vec.shape}"
        )

    tmp = frame.copy().reset_index(drop=True)

    for src, pd_name, lam in zip(
        SOURCE_COLS,
        PD_NAMES,
        lambda_vec
    ):
        tmp[pd_name] = compute_kernel_recursive(
            tmp[src].to_numpy(dtype=float),
            float(lam)
        )

        tmp[f"{pd_name}_lag1"] = tmp[pd_name]
        tmp[f"{pd_name}_lag5"] = tmp[pd_name].rolling(5).mean()
        tmp[f"{pd_name}_lag22"] = tmp[pd_name].rolling(22).mean()

    return tmp


def build_features(data_raw, lambda_vec, horizon):
    tmp = add_pd_features(
        data_raw,
        lambda_vec
    )

    tmp[f"RV_avg_h{horizon}"] = future_average_rv(
        tmp["RV"].to_numpy(dtype=float),
        horizon
    )

    tmp[f"forecast_end_DT_h{horizon}"] = (
        tmp["DT"].shift(-horizon)
    )

    tmp = tmp.dropna(
        subset=
            FEATURE_COLS
            + [
                f"RV_avg_h{horizon}",
                f"forecast_end_DT_h{horizon}",
            ]
    ).reset_index(drop=True)

    return tmp


# ============================================================
# 5. Fast purged chronological validation objective
# ============================================================

class WindowObjective:

    def __init__(self, frame, horizon):
        self.frame = frame
        self.horizon = horizon

        self.sources = [
            frame[c].to_numpy(dtype=float, copy=True)
            for c in SOURCE_COLS
        ]

        self.y_all = future_average_rv(
            frame["RV"].to_numpy(dtype=float),
            horizon
        )

        usable = np.arange(
            22,
            len(frame) - horizon
        )

        if len(usable) < 50:
            raise ValueError(
                "Not enough usable rows for chronological validation."
            )

        self.n_validation = int(
            np.ceil(
                len(usable)
                * VALIDATION_FRACTION
            )
        )

        split = (
            len(usable)
            - self.n_validation
        )

        # Purge h origins immediately before validation.
        self.fit_rows = usable[
            :max(0, split - horizon)
        ]

        self.validation_rows = usable[
            split:
        ]

        if (
            len(self.fit_rows) < 40
            or len(self.validation_rows) < 1
        ):
            raise ValueError(
                "Not enough observations after purged validation split."
            )

        assert (
            self.fit_rows[-1] + horizon
            < self.validation_rows[0]
        )

        self.objective_scale = max(
            float(
                np.mean(
                    self.y_all[self.validation_rows] ** 2
                )
            ),
            OBJECTIVE_SCALE_FLOOR,
        )

        self.blocks = [
            OrderedDict()
            for _ in SOURCE_COLS
        ]

        self.values = OrderedDict()
        self.evaluations = 0
        self.hits = 0


    def block(self, j, lam):
        cache = self.blocks[j]
        key = float(lam)

        if key in cache:
            self.hits += 1
            cache.move_to_end(key)
            return cache[key]

        s = compute_kernel_recursive(
            self.sources[j],
            key
        )

        series = pd.Series(s)

        block = np.column_stack([
            s,
            series.rolling(5).mean().to_numpy(),
            series.rolling(22).mean().to_numpy(),
        ])

        cache[key] = block

        if len(cache) > 64:
            cache.popitem(last=False)

        return block


    def evaluate(self, x):
        key = tuple(float(v) for v in x)

        if key in self.values:
            self.hits += 1
            self.values.move_to_end(key)
            return self.values[key]

        features = np.column_stack([
            self.block(j, v)
            for j, v in enumerate(key)
        ])

        fit = self.fit_rows
        val = self.validation_rows

        X_fit = np.column_stack([
            np.ones(len(fit)),
            features[fit],
        ])

        X_val = np.column_stack([
            np.ones(len(val)),
            features[val],
        ])

        y_fit = self.y_all[fit]
        y_val = self.y_all[val]

        if not all(
            np.all(np.isfinite(a))
            for a in (X_fit, X_val, y_fit, y_val)
        ):
            return 1e30

        pinv, _ = pinv_extended(X_fit)
        beta = pinv @ y_fit

        residual = (
            y_val
            - X_val @ beta
        )

        mse = float(
            np.mean(
                residual ** 2
            )
        )

        if not np.isfinite(mse):
            return 1e30

        self.evaluations += 1
        self.values[key] = mse

        if len(self.values) > 4096:
            self.values.popitem(last=False)

        return mse


def objective_validation_mse(lambda_vec, train_raw, horizon):
    global _FAST_CONTEXT

    x = np.asarray(lambda_vec, dtype=float)

    if (
        x.shape != (len(LAMBDA_NAMES),)
        or not np.all(np.isfinite(x))
        or np.any(x < LAMBDA_LOWER)
        or np.any(x > LAMBDA_UPPER)
    ):
        return 1e30

    if (
        _FAST_CONTEXT is None
        or _FAST_CONTEXT.frame is not train_raw
        or _FAST_CONTEXT.horizon != horizon
    ):
        _FAST_CONTEXT = WindowObjective(
            train_raw,
            horizon
        )

    return _FAST_CONTEXT.evaluate(x)


def objective_scaled(lambda_vec, train_raw, horizon):
    raw = objective_validation_mse(
        lambda_vec,
        train_raw,
        horizon
    )

    if not np.isfinite(raw) or raw >= 1e30:
        return 1e30

    return (
        raw
        / _FAST_CONTEXT.objective_scale
    )


# ============================================================
# 6. Powell search
# ============================================================

def generate_initial_starts():
    p = len(LAMBDA_NAMES)
    starts = []

    for value in [0.02, 0.1, 0.5, 2.0]:
        starts.append(
            np.full(
                p,
                value,
                dtype=float
            )
        )

    for j in range(p):
        x = np.full(
            p,
            0.5,
            dtype=float
        )
        x[j] = 0.05
        starts.append(x)

    for j in range(p):
        x = np.full(
            p,
            0.2,
            dtype=float
        )
        x[j] = 2.0
        starts.append(x)

    return starts


INITIAL_MULTI_STARTS = generate_initial_starts()


def run_single_powell(
    x0,
    train_raw,
    horizon,
    upper=LAMBDA_UPPER
):
    p = len(LAMBDA_NAMES)

    bounds = [
        (LAMBDA_LOWER, upper)
        for _ in range(p)
    ]

    x0 = np.clip(
        np.asarray(x0, dtype=float),
        LAMBDA_LOWER,
        upper
    )

    try:
        opt = minimize(
            objective_scaled,
            x0=x0,
            args=(train_raw, horizon),
            method="Powell",
            bounds=bounds,
            options={
                "maxiter": 500,
                "xtol": 1e-6,
                "ftol": 1e-10,
                "disp": False,
            }
        )

        candidates = [x0]

        if (
            opt.x is not None
            and np.all(np.isfinite(opt.x))
        ):
            candidates.append(
                np.clip(
                    np.asarray(opt.x, dtype=float),
                    LAMBDA_LOWER,
                    upper
                )
            )

        best_x = None
        best_mse = np.inf

        for x in candidates:
            mse = objective_validation_mse(
                x,
                train_raw,
                horizon
            )

            if (
                np.isfinite(mse)
                and mse < best_mse
            ):
                best_x = x.copy()
                best_mse = float(mse)

        if best_x is None:
            return None, np.inf

        return best_x, best_mse

    except Exception:
        return None, np.inf


def run_multiple_start_powell(
    train_raw,
    horizon,
    extra_starts=None,
    upper=LAMBDA_UPPER
):
    starts = []

    if extra_starts is not None:
        for x in extra_starts:
            if x is not None:
                starts.append(
                    np.asarray(x, dtype=float)
                )

    starts.extend([
        x.copy()
        for x in INITIAL_MULTI_STARTS
    ])

    unique = []
    seen = set()

    for x in starts:
        x = np.clip(
            x,
            LAMBDA_LOWER,
            upper
        )

        key = tuple(
            round(float(v), 10)
            for v in x
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(x)

    best_x = None
    best_mse = np.inf

    for x0 in unique:
        x, mse = run_single_powell(
            x0,
            train_raw,
            horizon,
            upper=upper
        )

        if (
            x is not None
            and np.isfinite(mse)
            and mse < best_mse
        ):
            best_x = x.copy()
            best_mse = float(mse)

    if best_x is None:
        raise RuntimeError(
            "Multiple-start Powell could not obtain valid lambdas."
        )

    return best_x, best_mse


def boundary_mask(x):
    x = np.asarray(x, dtype=float)

    return (
        x
        >= UPPER_BOUNDARY_THRESHOLD
    )


# ============================================================
# 7. Scalable 15-point grid audit
# ============================================================

def build_grid_audit_candidates(base_x):
    """
    Avoid an impractical 15^p Cartesian grid.

    Use:
    1) 15 diagonal points;
    2) 15-point coordinate profile for every lambda around base_x.

    This is the scalable coordinate/diagonal grid audit used for
    the component models with 3 or 4 lambdas.
    """
    p = len(LAMBDA_NAMES)

    axis = np.geomspace(
        LAMBDA_LOWER,
        LAMBDA_UPPER,
        GRID_SIZE
    )

    candidates = []

    # Diagonal
    for value in axis:
        candidates.append(
            np.full(
                p,
                value,
                dtype=float
            )
        )

    # Coordinate sweeps around current Powell solution
    for j in range(p):
        for value in axis:
            x = np.asarray(
                base_x,
                dtype=float
            ).copy()

            x[j] = value
            candidates.append(x)

    # Include base itself.
    candidates.append(
        np.asarray(
            base_x,
            dtype=float
        ).copy()
    )

    unique = []
    seen = set()

    for x in candidates:
        key = tuple(
            round(float(v), 10)
            for v in x
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(x)

    return unique


# ============================================================
# 8. 0.1% profile-equivalent interior boundary replacement
# ============================================================

def equivalence_allowance(best_mse):
    return max(
        PROFILE_ABS_TOL,
        PROFILE_REL_TOL
        * max(
            abs(float(best_mse)),
            OBJECTIVE_SCALE_FLOOR
        )
    )


def find_best_interior_solution(
    train_raw,
    horizon,
    unrestricted_x,
    audit_records
):
    starts = [
        np.clip(
            np.asarray(
                unrestricted_x,
                dtype=float
            ),
            LAMBDA_LOWER,
            INTERIOR_UPPER
        )
    ]

    interior_records = [
        (mse, x)
        for mse, x in audit_records
        if np.all(
            np.asarray(x, dtype=float)
            <= INTERIOR_UPPER
        )
    ]

    interior_records.sort(
        key=lambda z: z[0]
    )

    starts.extend([
        np.asarray(x, dtype=float).copy()
        for _, x in interior_records[
            :PROFILE_LOCAL_STARTS
        ]
    ])

    return run_multiple_start_powell(
        train_raw=train_raw,
        horizon=horizon,
        extra_starts=starts,
        upper=INTERIOR_UPPER
    )


def profile_shrink_boundary_coordinates(
    start_x,
    original_boundary,
    train_raw,
    horizon,
    mse_limit
):
    current_x = np.asarray(
        start_x,
        dtype=float
    ).copy()

    current_mse = float(
        objective_validation_mse(
            current_x,
            train_raw,
            horizon
        )
    )

    boundary_indices = np.where(
        original_boundary
    )[0]

    for j in boundary_indices:
        upper_here = min(
            float(current_x[j]),
            INTERIOR_UPPER
        )

        if upper_here <= LAMBDA_LOWER:
            continue

        grid_log = np.geomspace(
            LAMBDA_LOWER,
            upper_here,
            PROFILE_GRID_SIZE
        )

        grid_linear = np.linspace(
            LAMBDA_LOWER,
            upper_here,
            PROFILE_GRID_SIZE
        )

        values = np.unique(
            np.concatenate([
                grid_log,
                grid_linear,
                [upper_here]
            ])
        )

        values.sort()

        accepted_value = upper_here
        accepted_mse = current_mse

        # Smallest tested interior lambda that stays in the
        # 0.1% MSE-equivalence region.
        for value in values:
            x = current_x.copy()
            x[j] = float(value)

            mse = objective_validation_mse(
                x,
                train_raw,
                horizon
            )

            if (
                np.isfinite(mse)
                and mse <= mse_limit
            ):
                accepted_value = float(value)
                accepted_mse = float(mse)
                break

        current_x[j] = accepted_value
        current_mse = accepted_mse

    final_mse = float(
        objective_validation_mse(
            current_x,
            train_raw,
            horizon
        )
    )

    return current_x, final_mse


def estimate_lambda(
    train_raw,
    horizon,
    prev_lambda=None
):
    global _FAST_CONTEXT

    _FAST_CONTEXT = WindowObjective(
        train_raw,
        horizon
    )

    # --------------------------------------------------------
    # A. Base Powell
    # --------------------------------------------------------
    if prev_lambda is None:
        base_x, base_mse = run_multiple_start_powell(
            train_raw,
            horizon
        )
        base_mode = "initial_multiple_start_powell"
    else:
        warm = np.clip(
            np.asarray(
                prev_lambda,
                dtype=float
            ),
            LAMBDA_LOWER,
            LAMBDA_UPPER
        )

        base_x, base_mse = run_single_powell(
            warm,
            train_raw,
            horizon
        )

        if base_x is None:
            base_x, base_mse = run_multiple_start_powell(
                train_raw,
                horizon,
                extra_starts=[warm]
            )
            base_mode = "warm_start_multistart_fallback"
        else:
            base_mode = "daily_warm_start_powell"

    # --------------------------------------------------------
    # B. Scalable grid audit
    # --------------------------------------------------------
    best_x = np.asarray(
        base_x,
        dtype=float
    ).copy()

    best_mse = float(base_mse)
    search_source = "base_powell"

    audit_records = []

    for x in build_grid_audit_candidates(base_x):
        mse = objective_validation_mse(
            x,
            train_raw,
            horizon
        )

        if (
            np.isfinite(mse)
            and mse < 1e30
        ):
            audit_records.append(
                (float(mse), x.copy())
            )

            if mse < best_mse:
                best_x = x.copy()
                best_mse = float(mse)
                search_source = "grid_audit"

    audit_records.sort(
        key=lambda z: z[0]
    )

    for _, x0 in audit_records[
        :GRID_LOCAL_STARTS
    ]:
        x, mse = run_single_powell(
            x0,
            train_raw,
            horizon
        )

        if (
            x is not None
            and np.isfinite(mse)
            and mse < best_mse
        ):
            best_x = x.copy()
            best_mse = float(mse)
            search_source = "grid_local_powell"

    unrestricted_x = best_x.copy()
    unrestricted_mse = float(best_mse)

    original_boundary = boundary_mask(
        unrestricted_x
    )

    unrestricted_boundary_count = int(
        np.sum(original_boundary)
    )

    selected_x = unrestricted_x.copy()
    selected_mse = unrestricted_mse

    best_interior_mse = np.nan
    equivalent_interior_exists = False
    boundary_replaced = False
    replacement_source = "none"

    allowance = equivalence_allowance(
        unrestricted_mse
    )

    mse_limit = (
        unrestricted_mse
        + allowance
    )

    # --------------------------------------------------------
    # C. Only boundary cases enter interior-equivalence search
    # --------------------------------------------------------
    if unrestricted_boundary_count > 0:
        interior_x, interior_mse = (
            find_best_interior_solution(
                train_raw=train_raw,
                horizon=horizon,
                unrestricted_x=unrestricted_x,
                audit_records=audit_records
            )
        )

        best_interior_mse = float(
            interior_mse
        )

        if (
            np.isfinite(interior_mse)
            and interior_mse <= mse_limit
        ):
            equivalent_interior_exists = True

            profiled_x, profiled_mse = (
                profile_shrink_boundary_coordinates(
                    start_x=interior_x,
                    original_boundary=original_boundary,
                    train_raw=train_raw,
                    horizon=horizon,
                    mse_limit=mse_limit
                )
            )

            if (
                np.isfinite(profiled_mse)
                and profiled_mse <= mse_limit
                and np.all(
                    profiled_x
                    <= INTERIOR_UPPER
                )
            ):
                selected_x = profiled_x.copy()
                selected_mse = float(profiled_mse)
                replacement_source = (
                    "profile_equivalent_interior"
                )
            else:
                selected_x = np.asarray(
                    interior_x,
                    dtype=float
                ).copy()

                selected_mse = float(
                    interior_mse
                )

                replacement_source = (
                    "best_equivalent_interior"
                )

            boundary_replaced = True

    selected_boundary_count = int(
        np.sum(
            boundary_mask(selected_x)
        )
    )

    absolute_gap = float(
        selected_mse
        - unrestricted_mse
    )

    relative_gap = float(
        absolute_gap
        / max(
            abs(unrestricted_mse),
            OBJECTIVE_SCALE_FLOOR
        )
    )

    diag = {
        "mode":
            base_mode
            + "+"
            + search_source
            + (
                "+"
                + replacement_source
                if boundary_replaced
                else ""
            ),

        "search_source":
            search_source,

        "replacement_source":
            replacement_source,

        "pre_audit_validation_MSE":
            float(base_mse),

        "unrestricted_validation_MSE":
            unrestricted_mse,

        "best_interior_validation_MSE":
            (
                float(best_interior_mse)
                if np.isfinite(best_interior_mse)
                else np.nan
            ),

        "selected_validation_MSE":
            float(selected_mse),

        "selected_scaled_validation_MSE":
            float(
                selected_mse
                / _FAST_CONTEXT.objective_scale
            ),

        "objective_scale":
            float(
                _FAST_CONTEXT.objective_scale
            ),

        "equivalence_allowance":
            float(allowance),

        "equivalence_threshold":
            float(mse_limit),

        "absolute_MSE_gap":
            absolute_gap,

        "relative_MSE_gap":
            relative_gap,

        "unrestricted_boundary_count":
            unrestricted_boundary_count,

        "selected_boundary_count":
            selected_boundary_count,

        "equivalent_interior_exists":
            bool(equivalent_interior_exists),

        "boundary_replaced":
            bool(boundary_replaced),
    }

    for j, name in enumerate(LAMBDA_NAMES):
        diag[f"unrestricted_{name}"] = float(
            unrestricted_x[j]
        )

        diag[f"selected_{name}"] = float(
            selected_x[j]
        )

        diag[f"{name}_boundary"] = bool(
            selected_x[j]
            >= UPPER_BOUNDARY_THRESHOLD
        )

    return (
        np.asarray(
            selected_x,
            dtype=float
        ),
        float(selected_mse),
        diag
    )


# ============================================================
# 9. Final regression
# ============================================================


def fit_final_model(train_raw, lambda_vec, horizon):
    tmp = build_features(train_raw, lambda_vec, horizon)

    y = tmp[f"RV_avg_h{horizon}"]
    X = sm.add_constant(
        tmp[FEATURE_COLS],
        has_constant="add"
    )

    model = sm.OLS(y, X).fit()

    return model, None, tmp


def predict_final_model(model, scaler, X_test):
    pred = float(model.predict(X_test).iloc[0])
    return pred, np.nan, np.nan



def build_test_feature(
    train_raw,
    test_raw,
    lambda_vec
):
    combined = pd.concat(
        [train_raw, test_raw],
        axis=0,
        ignore_index=True
    )

    combined = add_pd_features(
        combined,
        lambda_vec
    )

    last_row = combined.iloc[
        [-1]
    ][FEATURE_COLS]

    if last_row.isna().any().any():
        raise RuntimeError(
            "Nonfinite test feature."
        )

    if "OLS" == "OLS":
        return sm.add_constant(
            last_row,
            has_constant="add"
        )

    return last_row


# ============================================================
# 10. Load merged 399006 data
# ============================================================

def load_data():
    # ---------- intraday ----------
    data_idx = pd.read_csv(
        INTRADAY_FILE,
        low_memory=False
    )

    required_intraday = {
        "time",
        "code",
        "close"
    }

    missing = (
        required_intraday
        - set(data_idx.columns)
    )

    if missing:
        raise ValueError(
            f"Intraday file is missing: {sorted(missing)}"
        )

    data_idx["code"] = (
        data_idx["code"]
        .astype(str)
        .str.strip()
    )

    data_idx = data_idx[
        data_idx["code"] == CODE
    ].copy()

    data_idx["datetime"] = pd.to_datetime(
        data_idx["time"],
        errors="coerce"
    )

    data_idx["close"] = pd.to_numeric(
        data_idx["close"],
        errors="coerce"
    )

    data_idx = (
        data_idx
        .dropna(
            subset=[
                "datetime",
                "close"
            ]
        )
        .copy()
    )

    data_idx = data_idx[
        data_idx["close"] > 0
    ].copy()

    # Filter first, same convention as supplied HAR-PD-RV.
    data_idx = data_idx[
        data_idx["datetime"]
        >= pd.Timestamp(START_DATE)
    ].copy()

    data_idx["date"] = (
        data_idx["datetime"]
        .dt.normalize()
    )

    data_idx = (
        data_idx
        .sort_values(
            [
                "date",
                "datetime"
            ]
        )
        .reset_index(drop=True)
    )

    component_data = (
        data_idx
        .groupby(
            "date",
            sort=True
        )
        .apply(
            calculate_daily_components
        )
        .reset_index()
        .rename(
            columns={
                "date": "DT"
            }
        )
    )

    component_data["DT"] = pd.to_datetime(
        component_data["DT"]
    )

    component_data = (
        component_data
        .dropna(
            subset=
                ["RV"]
                + COMPONENT_COLS
        )
        .sort_values("DT")
        .reset_index(drop=True)
    )

    # ---------- daily close -> R1T source ----------
    df_day = pd.read_csv(
        DAILY_FILE,
        low_memory=False
    )

    required_daily = {
        "time",
        "code",
        "close"
    }

    missing = (
        required_daily
        - set(df_day.columns)
    )

    if missing:
        raise ValueError(
            f"Daily file is missing: {sorted(missing)}"
        )

    df_day["code"] = (
        df_day["code"]
        .astype(str)
        .str.strip()
    )

    df_day = df_day[
        df_day["code"] == CODE
    ].copy()

    df_day["DT"] = pd.to_datetime(
        df_day["time"],
        errors="coerce"
    )

    df_day["close"] = pd.to_numeric(
        df_day["close"],
        errors="coerce"
    )

    df_day = (
        df_day
        .dropna(
            subset=[
                "DT",
                "close"
            ]
        )
        .copy()
    )

    df_day = df_day[
        df_day["close"] > 0
    ].copy()

    df_day = df_day[
        df_day["DT"]
        >= pd.Timestamp(START_DATE)
    ].copy()

    df_day = (
        df_day
        .sort_values("DT")
        .drop_duplicates(
            subset=["DT"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    df_day["returns"] = (
        df_day["close"]
        / df_day["close"].shift(1)
        - 1.0
    )

    returns_df = (
        df_day[
            [
                "DT",
                "returns"
            ]
        ]
        .dropna(
            subset=["returns"]
        )
        .copy()
    )

    data = pd.merge(
        component_data,
        returns_df,
        on="DT",
        how="inner"
    )

    data = (
        data
        .dropna(
            subset=
                ["DT", "RV", "returns"]
                + COMPONENT_COLS
        )
        .sort_values("DT")
        .reset_index(drop=True)
    )

    # Decomposition diagnostic
    data["RV_check"] = (
        data[COMPONENT_COLS]
        .sum(axis=1)
    )

    data["decomp_error"] = np.abs(
        data["RV"]
        - data["RV_check"]
    )

    print("=" * 105)
    print(f"{MODEL_LABEL} | BASE DATA")
    print("=" * 105)
    print(f"Code             : {CODE}")
    print(f"Configured start : {START_DATE}")
    print(f"Actual start     : {data['DT'].min()}")
    print(f"Actual end       : {data['DT'].max()}")
    print(f"Daily N          : {len(data)}")
    print(
        "Max decomposition error:",
        data["decomp_error"].max()
    )
    print("\nFirst 5 rows:")
    print(data.head())
    print("=" * 105)

    return data


# ============================================================
# 11. Common-origin fixed-length rolling OOS
# ============================================================

def rolling_forecast(
    data,
    horizon,
    forecast_window=FORECAST_WINDOW,
    common_max_horizon=COMMON_MAX_HORIZON
):
    data = (
        data
        .copy()
        .dropna(
            subset=
                ["DT", "RV", "returns"]
                + COMPONENT_COLS
        )
        .sort_values("DT")
        .reset_index(drop=True)
    )

    if data["DT"].duplicated().any():
        raise ValueError(
            "Duplicate daily dates detected."
        )

    n_total = len(data)

    # Same 1000 origins for H1/H5/H22, anchored by H22.
    last_common_origin_idx = (
        n_total
        - common_max_horizon
        - 1
    )

    first_common_origin_idx = (
        last_common_origin_idx
        - forecast_window
        + 1
    )

    if first_common_origin_idx <= 100:
        raise ValueError(
            "Too little history for common-origin rolling."
        )

    train_window = first_common_origin_idx

    print("\n" + "=" * 110)
    print(
        f"{MODEL_LABEL} | "
        f"h={horizon}"
    )
    print("=" * 110)
    print(
        f"Total daily sample N            : {n_total}"
    )
    print(
        f"Fixed raw rolling train window  : {train_window}"
    )
    print(
        f"Common OOS observations         : {forecast_window}"
    )
    print(
        f"Common OOS origin start         : "
        f"{data['DT'].iloc[first_common_origin_idx]}"
    )
    print(
        f"Common OOS origin end           : "
        f"{data['DT'].iloc[last_common_origin_idx]}"
    )
    print(
        "Lambda validation fraction      : 5%"
    )
    print(
        "Boundary handling               : "
        "0.1% profile-equivalent interior replacement"
    )
    print("=" * 110)

    rows = []
    prev_lambda = None

    for i in range(forecast_window):
        started = perf_counter()

        origin_idx = (
            first_common_origin_idx
            + i
        )

        train_start_idx = (
            origin_idx
            - train_window
        )

        train_raw = data.iloc[
            train_start_idx:
            origin_idx
        ].copy()

        test_raw = data.iloc[
            origin_idx:
            origin_idx + 1
        ].copy()

        if (
            len(train_raw) != train_window
            or len(test_raw) != 1
        ):
            raise RuntimeError(
                f"Rolling indexing error at i={i}"
            )

        actual_window = data.iloc[
            origin_idx + 1:
            origin_idx + horizon + 1
        ].copy()

        if len(actual_window) != horizon:
            raise RuntimeError(
                f"h={horizon}, i={i}: "
                "incomplete actual window"
            )

        actual = float(
            actual_window["RV"].mean()
        )

        forecast_start_date = (
            actual_window["DT"].iloc[0]
        )

        forecast_end_date = (
            actual_window["DT"].iloc[-1]
        )

        # ----------------------------------------------------
        # Lambda selection
        # ----------------------------------------------------
        lambda_hat, selected_objective, diag = (
            estimate_lambda(
                train_raw=train_raw,
                horizon=horizon,
                prev_lambda=prev_lambda
            )
        )

        prev_lambda = lambda_hat.copy()

        # ----------------------------------------------------
        # Final OLS or LassoCV
        # ----------------------------------------------------
        model, scaler, train_model_data = (
            fit_final_model(
                train_raw=train_raw,
                lambda_vec=lambda_hat,
                horizon=horizon
            )
        )

        X_test = build_test_feature(
            train_raw=train_raw,
            test_raw=test_raw,
            lambda_vec=lambda_hat
        )

        pred_raw, lasso_alpha, lasso_nonzero = (
            predict_final_model(
                model,
                scaler,
                X_test
            )
        )

        pred = (
            1e-5
            if (
                not np.isfinite(pred_raw)
                or pred_raw < 0
            )
            else pred_raw
        )

        row = {
            "origin_DT":
                test_raw["DT"].iloc[0],

            "forecast_start_DT":
                forecast_start_date,

            "forecast_end_DT":
                forecast_end_date,

            "train_start_DT":
                train_raw["DT"].iloc[0],

            "train_end_DT":
                train_raw["DT"].iloc[-1],

            "raw_train_n":
                len(train_raw),

            "usable_train_n":
                len(train_model_data),

            "RV_actual":
                actual,

            "RV_pred_raw":
                pred_raw,

            "RV_pred":
                pred,

            "selected_validation_MSE":
                diag["selected_validation_MSE"],

            "unrestricted_validation_MSE":
                diag["unrestricted_validation_MSE"],

            "best_interior_validation_MSE":
                diag["best_interior_validation_MSE"],

            "relative_MSE_gap":
                diag["relative_MSE_gap"],

            "equivalence_threshold":
                diag["equivalence_threshold"],

            "equivalent_interior_exists":
                diag["equivalent_interior_exists"],

            "boundary_replaced":
                diag["boundary_replaced"],

            "unrestricted_boundary_count":
                diag["unrestricted_boundary_count"],

            "selected_boundary_count":
                diag["selected_boundary_count"],

            "replacement_source":
                diag["replacement_source"],

            "optimization_mode":
                diag["mode"],

            "validation_n":
                len(_FAST_CONTEXT.validation_rows),

            "purged_n":
                horizon,
        }

        if "OLS" == "LASSO":
            row["lasso_alpha"] = lasso_alpha
            row["lasso_nonzero"] = lasso_nonzero

        for j, name in enumerate(LAMBDA_NAMES):
            row[
                f"unrestricted_{name}_h{horizon}"
            ] = diag[
                f"unrestricted_{name}"
            ]

            row[
                f"{name}_h{horizon}"
            ] = float(
                lambda_hat[j]
            )

            row[
                f"{name}_boundary"
            ] = diag[
                f"{name}_boundary"
            ]

        rows.append(row)

        if (i + 1) % 50 == 0:
            lambda_text = " | ".join([
                f"{name}={lambda_hat[j]:.6f}"
                for j, name
                in enumerate(LAMBDA_NAMES)
            ])

            extra = (
                f" | alpha={lasso_alpha:.8g}"
                if "OLS" == "LASSO"
                else ""
            )

            print(
                f"{MODEL_LABEL} | "
                f"h={horizon} | "
                f"{i + 1}/{forecast_window} | "
                f"{lambda_text} | "
                f"validation_MSE={selected_objective:.10e} | "
                f"replaced={diag['boundary_replaced']}"
                f"{extra} | "
                f"seconds={perf_counter() - started:.2f}",
                flush=True
            )

    results = pd.DataFrame(rows)

    actuals = results[
        "RV_actual"
    ].to_numpy(dtype=float)

    preds = results[
        "RV_pred"
    ].to_numpy(dtype=float)

    mse = mean_squared_error(
        actuals,
        preds
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        actuals,
        preds
    )

    r2 = r2_score(
        actuals,
        preds
    )

    valid = (
        np.isfinite(actuals)
        & np.isfinite(preds)
        & (actuals > 0)
        & (preds > 0)
    )

    qlike = (
        float(
            np.mean(
                np.log(preds[valid])
                + actuals[valid]
                / preds[valid]
            )
        )
        if np.any(valid)
        else np.nan
    )

    print("\n" + "=" * 110)
    print(
        f"{MODEL_LABEL} | "
        f"OOS RESULTS | h={horizon}"
    )
    print("=" * 110)
    print(
        f"OOS observations         : {len(results)}"
    )
    print(
        f"Boundary replaced origins: "
        f"{int(results['boundary_replaced'].sum())}"
    )
    print(
        f"Remaining boundary origins: "
        f"{int((results['selected_boundary_count'] > 0).sum())}"
    )
    print(f"MSE   : {mse:.12e}")
    print(f"RMSE  : {rmse:.12e}")
    print(f"MAE   : {mae:.12e}")
    print(f"R2    : {r2:.8f}")
    print(f"QLIKE : {qlike:.12e}")
    print("=" * 110)

    return (
        results,
        float(mse),
        float(qlike)
    )


# ============================================================
# 12. Run
# ============================================================

if __name__ == "__main__":
    print("\nACTUAL RUNNING FILE:")
    print(os.path.abspath(__file__))
    print(
        "VERSION: "
        f"{MODEL_LABEL} | "
        "5% validation | "
        "lambda upper 5 | "
        "0.1% profile-equivalent boundary replacement"
    )

    data = load_data()

    output_dir = Path(
        __file__
    ).resolve().parent

    summaries = []

    for h in HORIZONS:
        result, mse, qlike = rolling_forecast(
            data=data,
            horizon=h,
            forecast_window=FORECAST_WINDOW,
            common_max_horizon=COMMON_MAX_HORIZON
        )

        output_file = (
            output_dir
            / f"{FILE_PREFIX}-H{h}.csv"
        )

        result.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"Saved: {output_file}"
        )

        summaries.append({
            "Horizon": h,
            "MSE": mse,
            "QLIKE": qlike
        })

    print("\n" + "=" * 95)
    print(
        f"{MODEL_LABEL} FINAL LOSS SUMMARY"
    )
    print("=" * 95)
    print(
        pd.DataFrame(
            summaries
        ).to_string(index=False)
    )
    print("=" * 95)
