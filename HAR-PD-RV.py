

import os
import warnings
from collections import OrderedDict
from time import perf_counter

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.signal import lfilter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tools.tools import pinv_extended

warnings.filterwarnings("ignore")

# ============================================================
# 0. Settings
# ============================================================
INTRADAY_FILE = r"data_idx.csv"
DAILY_FILE = r"day_data.csv"
CODE = "data"
START_DATE = "date"

ESTIMATION_WINDOW = 1500
FORECAST_WINDOW = 300
HORIZONS = [1, 5, 22]

VALIDATION_FRACTION = 0.01

LAMBDA_LOWER = 0.0001
LAMBDA_UPPER = 5.0
BOUNDARY_TOL = 0.01
UPPER_BOUNDARY_THRESHOLD = LAMBDA_UPPER - BOUNDARY_TOL

GRID_LOW_SIZE = 8
GRID_HIGH_SIZE = 8
GRID_LOCAL_STARTS = 3

TOP3_MIN_LOG_DISTANCE = float(np.log(1.5))

MODEL_LABEL = "HAR-PD-RV"
FILE_PREFIX = "HAR-PD-RV"

LAMBDA_NAMES = ["lambda_R1T", "lambda_RVK"]
FEATURE_COLS = [
    "R1T_lag1", "R1T_lag5", "R1T_lag22",
    "RVK_lag1", "RVK_lag5", "RVK_lag22",
]

def parse_datetime(series):
    raw = series.astype(str).str.strip()

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )

    numeric_mask = raw.str.fullmatch(
        r"[0-9]+(?:\.[0-9]+)?",
        na=False,
    )

    if numeric_mask.any():
        values = pd.to_numeric(
            raw.loc[numeric_mask],
            errors="coerce",
        )

        ms_mask = values.notna() & (values > 1e11)
        if ms_mask.any():
            parsed = pd.to_datetime(
                values.loc[ms_mask],
                unit="ms",
                origin="unix",
                utc=True,
                errors="coerce",
            )
            parsed = (
                parsed.dt.tz_convert("Asia/Shanghai")
                .dt.tz_localize(None)
            )
            result.loc[parsed.index] = parsed

        sec_mask = values.notna() & (values > 1e8) & ~ms_mask
        if sec_mask.any():
            parsed = pd.to_datetime(
                values.loc[sec_mask],
                unit="s",
                origin="unix",
                utc=True,
                errors="coerce",
            )
            parsed = (
                parsed.dt.tz_convert("Asia/Shanghai")
                .dt.tz_localize(None)
            )
            result.loc[parsed.index] = parsed

    remaining = result.isna()
    if remaining.any():
        strings = (
            raw.loc[remaining]
            .str.replace("/", "-", regex=False)
            .str.replace("T", " ", regex=False)
            .str.strip()
        )

        try:
            parsed = pd.to_datetime(
                strings,
                format="mixed",
                errors="coerce",
            )
        except TypeError:
            parsed = pd.to_datetime(
                strings,
                errors="coerce",
            )

        try:
            if getattr(parsed.dt, "tz", None) is not None:
                parsed = (
                    parsed.dt.tz_convert("Asia/Shanghai")
                    .dt.tz_localize(None)
                )
        except Exception:
            pass

        result.loc[parsed.index] = parsed

    return result


def calculate_daily_rv(group):
    group = group.sort_values("datetime").copy()
    price = group["close"].to_numpy(dtype=float)

    if len(price) < 3 or np.any(~np.isfinite(price)) or np.any(price <= 0):
        return pd.Series({"RV": np.nan})

    log_ret = np.diff(np.log(price))
    if len(log_ret) < 2:
        return pd.Series({"RV": np.nan})

    return pd.Series({"RV": float(np.sum(log_ret ** 2))})


def compute_kernel_recursive(x, lam):
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)

    if n == 0:
        return out

    decay = np.exp(-lam)
    b = [lam]
    a = [1.0, -decay]

    # Current x_t is included.
    raw_val = lfilter(b, a, x)
    raw_w = lfilter(b, a, np.ones(n))

    valid = raw_w > 1e-15
    out[valid] = raw_val[valid] / raw_w[valid]
    return out


def future_average_rv(rv_values, horizon):
    rv_values = np.asarray(rv_values, dtype=float)
    n = len(rv_values)
    out = np.full(n, np.nan)

    for i in range(n):
        if i + horizon < n:
            out[i] = np.mean(rv_values[i + 1:i + horizon + 1])

    return out


def add_future_average_target(data_raw, horizon):
    tmp = data_raw.copy().sort_values("DT").reset_index(drop=True)
    target_col = f"RV_avg_h{horizon}"
    end_col = f"forecast_end_DT_h{horizon}"

    tmp[target_col] = future_average_rv(tmp["RV"].values, horizon)
    tmp[end_col] = tmp["DT"].shift(-horizon)

    return tmp.dropna(subset=[target_col, end_col]).reset_index(drop=True)


def pd_state_tail(
    state_raw,
    package_raw,
    source_col,
    lam,
):
    state_raw = (
        state_raw
        .copy()
        .sort_values("DT")
        .reset_index(drop=True)
    )

    package_raw = (
        package_raw
        .copy()
        .sort_values("DT")
        .reset_index(drop=True)
    )

    if len(state_raw) < len(package_raw):
        raise ValueError(
            "state_raw cannot be shorter than package_raw."
        )

    if (
        state_raw["DT"].iloc[-1]
        != package_raw["DT"].iloc[-1]
    ):
        raise RuntimeError(
            "state_raw and package_raw must end at the same origin date."
        )

    offset = (
        len(state_raw)
        -
        len(package_raw)
    )

    tail_dates = (
        state_raw["DT"]
        .iloc[offset:]
        .reset_index(drop=True)
    )

    if not tail_dates.equals(
        package_raw["DT"]
    ):
        raise RuntimeError(
            "package_raw must be the trailing block of state_raw."
        )

    full_state = compute_kernel_recursive(
        state_raw[source_col].to_numpy(
            dtype=float
        ),
        float(lam),
    )

    package_state = (
        full_state[
            offset:
        ]
    )

    if len(package_state) != len(package_raw):
        raise RuntimeError(
            "PD-state slicing length mismatch."
        )

    return np.asarray(
        package_state,
        dtype=float,
    )


def build_features(
    package_raw,
    state_raw,
    lambda_vec,
    horizon,
):
    lam_R1T, lam_RVK = np.asarray(
        lambda_vec,
        dtype=float,
    )

    tmp = (
        package_raw
        .copy()
        .sort_values("DT")
        .reset_index(drop=True)
    )

    tmp["R1T"] = pd_state_tail(
        state_raw=state_raw,
        package_raw=tmp,
        source_col="returns",
        lam=lam_R1T,
    )

    tmp["RVK"] = pd_state_tail(
        state_raw=state_raw,
        package_raw=tmp,
        source_col="RV",
        lam=lam_RVK,
    )

    tmp["R1T_lag1"] = tmp["R1T"]
    tmp["R1T_lag5"] = (
        tmp["R1T"]
        .rolling(window=5)
        .mean()
    )
    tmp["R1T_lag22"] = (
        tmp["R1T"]
        .rolling(window=22)
        .mean()
    )

    tmp["RVK_lag1"] = tmp["RVK"]
    tmp["RVK_lag5"] = (
        tmp["RVK"]
        .rolling(window=5)
        .mean()
    )
    tmp["RVK_lag22"] = (
        tmp["RVK"]
        .rolling(window=22)
        .mean()
    )

    tmp[
        f"RV_avg_h{horizon}"
    ] = future_average_rv(
        tmp["RV"].to_numpy(
            dtype=float
        ),
        horizon,
    )

    tmp = (
        tmp
        .dropna(
            subset=(
                FEATURE_COLS
                + [
                    f"RV_avg_h{horizon}"
                ]
            )
        )
        .reset_index(drop=True)
    )

    return tmp


_FAST_CONTEXT = None


class WindowObjective:
    def __init__(
        self,
        package_frame,
        state_frame,
        horizon,
    ):
        self.package_frame = package_frame
        self.state_frame = state_frame
        self.horizon = horizon

        self.package_frame = (
            self.package_frame
            .copy()
            .sort_values("DT")
            .reset_index(drop=True)
        )

        self.state_frame = (
            self.state_frame
            .copy()
            .sort_values("DT")
            .reset_index(drop=True)
        )

        if len(
            self.state_frame
        ) < len(
            self.package_frame
        ):
            raise ValueError(
                "state_frame cannot be shorter than package_frame."
            )

        if (
            self.state_frame["DT"].iloc[-1]
            != self.package_frame["DT"].iloc[-1]
        ):
            raise RuntimeError(
                "state_frame and package_frame must end at the same origin."
            )

        self.state_offset = (
            len(self.state_frame)
            -
            len(self.package_frame)
        )

        state_tail_dates = (
            self.state_frame["DT"]
            .iloc[
                self.state_offset:
            ]
            .reset_index(drop=True)
        )

        if not state_tail_dates.equals(
            self.package_frame["DT"]
        ):
            raise RuntimeError(
                "package_frame must be the trailing block of state_frame."
            )

        self.sources_state = [
            self.state_frame[
                "returns"
            ].to_numpy(
                dtype=float,
                copy=True,
            ),
            self.state_frame[
                "RV"
            ].to_numpy(
                dtype=float,
                copy=True,
            ),
        ]

        self.y_all = future_average_rv(
            self.package_frame[
                "RV"
            ].to_numpy(
                dtype=float
            ),
            horizon,
        )

        # HAR 22-day feature is still package-local.
        usable = np.arange(
            21,
            len(self.package_frame)
            - horizon,
        )

        if len(usable) < 50:
            raise ValueError(
                "Not enough usable observations "
                "for chronological validation."
            )

        self.n_validation = int(
            np.ceil(
                len(usable)
                * VALIDATION_FRACTION
            )
        )

        split = (
            len(usable)
            -
            self.n_validation
        )

        # h-origin purge prevents target-window overlap.
        self.fit_rows = usable[
            :
            max(
                0,
                split - horizon,
            )
        ]

        self.validation_rows = usable[
            split:
        ]

        if (
            len(self.fit_rows) < 40
            or len(self.validation_rows) < 1
        ):
            raise ValueError(
                "Not enough observations for "
                "purged 1% chronological validation."
            )

        assert (
            self.fit_rows[-1]
            + horizon
            <
            self.validation_rows[0]
        )

        self.blocks = [
            OrderedDict()
            for _ in range(2)
        ]

        self.values = OrderedDict()
        self.evaluations = 0
        self.hits = 0

    def block(
        self,
        j,
        lam,
    ):
        cache = self.blocks[j]
        key = float(lam)

        if key in cache:
            self.hits += 1
            cache.move_to_end(key)
            return cache[key]

        # Compute recursive state over ALL available history.
        s_full = compute_kernel_recursive(
            self.sources_state[j],
            key,
        )

        # Keep only the formal fixed 1500-day package.
        s = np.asarray(
            s_full[
                self.state_offset:
            ],
            dtype=float,
        )

        if len(s) != len(
            self.package_frame
        ):
            raise RuntimeError(
                "Objective PD-state slicing length mismatch."
            )

        # 5/22 averages remain package-local to preserve
        # the formal HAR estimation rows.
        series = pd.Series(s)

        block = np.column_stack([
            s,
            series
            .rolling(5)
            .mean()
            .to_numpy(),
            series
            .rolling(22)
            .mean()
            .to_numpy(),
        ])

        cache[key] = block

        if len(cache) > 96:
            cache.popitem(
                last=False
            )

        return block

    def evaluate(
        self,
        x,
    ):
        key = tuple(
            float(v)
            for v in x
        )

        if key in self.values:
            self.hits += 1
            self.values.move_to_end(key)
            return self.values[key]

        features = np.column_stack([
            self.block(
                j,
                v,
            )
            for j, v
            in enumerate(key)
        ])

        fit = self.fit_rows
        val = self.validation_rows

        X_fit = np.column_stack([
            np.ones(
                len(fit)
            ),
            features[
                fit
            ],
        ])

        X_val = np.column_stack([
            np.ones(
                len(val)
            ),
            features[
                val
            ],
        ])

        y_fit = self.y_all[
            fit
        ]

        y_val = self.y_all[
            val
        ]

        arrays = (
            X_fit,
            X_val,
            y_fit,
            y_val,
        )

        if not all(
            np.all(
                np.isfinite(a)
            )
            for a in arrays
        ):
            return 1e30

        pinv, _ = pinv_extended(
            X_fit
        )

        beta = (
            pinv
            @ y_fit
        )

        residual = (
            y_val
            -
            X_val
            @ beta
        )

        mse = float(
            np.mean(
                residual ** 2
            )
        )

        if not np.isfinite(
            mse
        ):
            return 1e30

        self.evaluations += 1

        self.values[
            key
        ] = mse

        if len(
            self.values
        ) > 4096:
            self.values.popitem(
                last=False
            )

        return mse


def objective_validation_mse(
    lambda_vec,
    package_raw,
    state_raw,
    horizon,
):
    global _FAST_CONTEXT

    x = np.asarray(
        lambda_vec,
        dtype=float,
    )

    if (
        x.shape != (2,)
        or not np.all(
            np.isfinite(x)
        )
        or np.any(
            x < LAMBDA_LOWER
        )
        or np.any(
            x > LAMBDA_UPPER
        )
    ):
        return 1e30

    if (
        _FAST_CONTEXT is None
        or _FAST_CONTEXT.horizon
        != horizon
    ):
        _FAST_CONTEXT = WindowObjective(
            package_frame=package_raw,
            state_frame=state_raw,
            horizon=horizon,
        )

    return _FAST_CONTEXT.evaluate(
        x
    )


def make_lambda_axis():

    low = np.geomspace(
        LAMBDA_LOWER,
        0.5,
        GRID_LOW_SIZE,
    )

    high = np.linspace(
        0.5,
        LAMBDA_UPPER,
        GRID_HIGH_SIZE,
    )

    axis = np.unique(
        np.concatenate([
            low,
            high,
        ])
    )

    if len(axis) != 15:
        raise RuntimeError(
            f"Hybrid grid must have 15 unique points; "
            f"got {len(axis)}."
        )

    if not np.isclose(
        axis[0],
        LAMBDA_LOWER,
    ):
        raise RuntimeError(
            "Hybrid grid does not include lower bound."
        )

    if not np.isclose(
        axis[-1],
        LAMBDA_UPPER,
    ):
        raise RuntimeError(
            "Hybrid grid does not include upper bound."
        )

    return axis


def run_single_powell(
    x0,
    package_raw,
    state_raw,
    horizon,
):
    bounds = [
        (
            LAMBDA_LOWER,
            LAMBDA_UPPER,
        )
    ] * 2

    x0 = np.clip(
        np.asarray(
            x0,
            dtype=float,
        ),
        LAMBDA_LOWER,
        LAMBDA_UPPER,
    )

    try:
        opt = minimize(
            objective_validation_mse,
            x0=x0,
            args=(
                package_raw,
                state_raw,
                horizon,
            ),
            method="Powell",
            bounds=bounds,
            options={
                "maxiter": 500,
                "xtol": 1e-6,
                "ftol": 1e-10,
                "disp": False,
            },
        )

        candidates = [
            x0
        ]

        if (
            opt.x is not None
            and np.all(
                np.isfinite(
                    opt.x
                )
            )
        ):
            candidates.append(
                np.clip(
                    np.asarray(
                        opt.x,
                        dtype=float,
                    ),
                    LAMBDA_LOWER,
                    LAMBDA_UPPER,
                )
            )

        best_x = None
        best_fun = np.inf

        for x in candidates:
            f = objective_validation_mse(
                x,
                package_raw,
                state_raw,
                horizon,
            )

            if (
                np.isfinite(f)
                and f < best_fun
            ):
                best_x = (
                    x.copy()
                )
                best_fun = float(
                    f
                )

        if best_x is None:
            return (
                None,
                np.inf,
            )

        return (
            best_x,
            best_fun,
        )

    except Exception:
        return (
            None,
            np.inf,
        )


def boundary_mask(
    x,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    return (
        x
        >=
        UPPER_BOUNDARY_THRESHOLD
    )


def log_lambda_distance(
    x,
    y,
):
    x = np.maximum(
        np.asarray(
            x,
            dtype=float,
        ),
        LAMBDA_LOWER,
    )

    y = np.maximum(
        np.asarray(
            y,
            dtype=float,
        ),
        LAMBDA_LOWER,
    )

    return float(
        np.linalg.norm(
            np.log(x)
            -
            np.log(y)
        )
    )


def select_diverse_top_starts(
    records,
    n_starts=GRID_LOCAL_STARTS,
):
    """
    Greedy selection:
    - always keep the global grid winner;
    - add subsequent low-MSE grid candidates only if they are
      sufficiently separated from every already selected start
      in log-lambda space;
    - if fewer than n_starts satisfy the separation rule,
      fill the remainder with the best still-unused grid points.
    """
    selected = []

    for mse, x in records:

        if not selected:
            selected.append(
                (
                    float(mse),
                    x.copy(),
                )
            )

        else:
            sufficiently_far = all(
                log_lambda_distance(
                    x,
                    chosen_x,
                )
                >=
                TOP3_MIN_LOG_DISTANCE
                for _, chosen_x
                in selected
            )

            if sufficiently_far:
                selected.append(
                    (
                        float(mse),
                        x.copy(),
                    )
                )

        if len(selected) >= n_starts:
            break

    if len(selected) < n_starts:

        used = {
            tuple(
                np.round(
                    x,
                    12,
                )
            )
            for _, x
            in selected
        }

        for mse, x in records:
            key = tuple(
                np.round(
                    x,
                    12,
                )
            )

            if key in used:
                continue

            selected.append(
                (
                    float(mse),
                    x.copy(),
                )
            )

            used.add(
                key
            )

            if len(
                selected
            ) >= n_starts:
                break

    return selected


# ============================================================
# 7. Lambda estimation
#
# Every origin is estimated independently:
#
#     fresh 15x15 hybrid grid
#         ->
#     3 diversified best grid starts
#         ->
#     3 bounded Powell refinements
#
# NO warm start.
# NO previous-origin lambda.
# ============================================================

def estimate_lambda(
    package_raw,
    state_raw,
    horizon,
):
    global _FAST_CONTEXT

    _FAST_CONTEXT = WindowObjective(
        package_frame=package_raw,
        state_frame=state_raw,
        horizon=horizon,
    )

    axis = make_lambda_axis()

    records = []

    for a in axis:
        for b in axis:

            x = np.array(
                [
                    a,
                    b,
                ],
                dtype=float,
            )

            mse = objective_validation_mse(
                x,
                package_raw,
                state_raw,
                horizon,
            )

            if (
                np.isfinite(
                    mse
                )
                and mse < 1e30
            ):
                records.append(
                    (
                        float(mse),
                        x.copy(),
                    )
                )

    if not records:
        raise RuntimeError(
            "No valid hybrid-grid candidate "
            "in current rolling window."
        )

    records.sort(
        key=lambda r: r[0]
    )

    # Grid winner is always retained as a final candidate.
    grid_best_mse = float(
        records[0][0]
    )

    grid_best_x = (
        records[0][1]
        .copy()
    )

    best_mse = grid_best_mse
    best_x = (
        grid_best_x
        .copy()
    )

    source = "hybrid_grid"

    diverse_starts = (
        select_diverse_top_starts(
            records,
            n_starts=GRID_LOCAL_STARTS,
        )
    )

    refined_records = []

    for rank, (
        start_mse,
        x0,
    ) in enumerate(
        diverse_starts,
        start=1,
    ):

        x_refined, mse_refined = (
            run_single_powell(
                x0=x0,
                package_raw=package_raw,
                state_raw=state_raw,
                horizon=horizon,
            )
        )

        if (
            x_refined is not None
            and np.isfinite(
                mse_refined
            )
            and mse_refined < 1e30
        ):
            refined_records.append({
                "rank":
                    rank,

                "start_MSE":
                    float(
                        start_mse
                    ),

                "start_x":
                    x0.copy(),

                "refined_MSE":
                    float(
                        mse_refined
                    ),

                "refined_x":
                    x_refined.copy(),
            })

            if mse_refined < best_mse:

                best_mse = float(
                    mse_refined
                )

                best_x = (
                    x_refined
                    .copy()
                )

                source = (
                    f"diverse_top{rank}_powell"
                )

    interior_grid_mse = [
        mse
        for mse, x
        in records
        if np.all(
            x
            <
            UPPER_BOUNDARY_THRESHOLD
        )
    ]

    # Best-vs-second-best grid gap as a weak-identification diagnostic.
    if len(records) >= 2:
        second_grid_mse = float(
            records[1][0]
        )
        relative_grid_gap = float(
            (
                second_grid_mse
                -
                grid_best_mse
            )
            /
            max(
                abs(
                    grid_best_mse
                ),
                1e-30,
            )
        )
    else:
        second_grid_mse = np.nan
        relative_grid_gap = np.nan

    # Diversity diagnostics for selected starts.
    selected_start_1 = (
        diverse_starts[0][1]
        if len(diverse_starts) > 0
        else np.array(
            [
                np.nan,
                np.nan,
            ]
        )
    )

    selected_start_2 = (
        diverse_starts[1][1]
        if len(diverse_starts) > 1
        else np.array(
            [
                np.nan,
                np.nan,
            ]
        )
    )

    selected_start_3 = (
        diverse_starts[2][1]
        if len(diverse_starts) > 2
        else np.array(
            [
                np.nan,
                np.nan,
            ]
        )
    )

    diag = {
        "mode":
            "state_history+hybrid_grid+diverse_top3_powell",

        "search_source":
            source,

        "grid_best_validation_MSE":
            float(
                grid_best_mse
            ),

        "grid_second_validation_MSE":
            second_grid_mse,

        "grid_relative_best_second_gap":
            relative_grid_gap,

        "grid_best_lambda_R1T":
            float(
                grid_best_x[0]
            ),

        "grid_best_lambda_RVK":
            float(
                grid_best_x[1]
            ),

        "grid_interior_best_validation_MSE":
            (
                float(
                    min(
                        interior_grid_mse
                    )
                )
                if interior_grid_mse
                else np.nan
            ),

        "selected_validation_MSE":
            float(
                best_mse
            ),

        "audit_MSE_improvement":
            float(
                grid_best_mse
                -
                best_mse
            ),

        "boundary_count":
            int(
                np.sum(
                    boundary_mask(
                        best_x
                    )
                )
            ),

        "lambda_R1T_boundary":
            bool(
                best_x[0]
                >=
                UPPER_BOUNDARY_THRESHOLD
            ),

        "lambda_RVK_boundary":
            bool(
                best_x[1]
                >=
                UPPER_BOUNDARY_THRESHOLD
            ),

        "selected_lambda_R1T":
            float(
                best_x[0]
            ),

        "selected_lambda_RVK":
            float(
                best_x[1]
            ),

        "powell_refinements_attempted":
            int(
                len(
                    diverse_starts
                )
            ),

        "powell_refinements_valid":
            int(
                len(
                    refined_records
                )
            ),

        "start1_R1T":
            float(
                selected_start_1[0]
            ),

        "start1_RVK":
            float(
                selected_start_1[1]
            ),

        "start2_R1T":
            float(
                selected_start_2[0]
            ),

        "start2_RVK":
            float(
                selected_start_2[1]
            ),

        "start3_R1T":
            float(
                selected_start_3[0]
            ),

        "start3_RVK":
            float(
                selected_start_3[1]
            ),
    }

    return (
        np.asarray(
            best_x,
            dtype=float,
        ),
        float(
            best_mse
        ),
        diag,
    )


# ============================================================
# 8. Final OLS using only formal package rows
# ============================================================

def fit_model(
    package_raw,
    state_raw,
    lambda_vec,
    horizon,
):
    tmp = build_features(
        package_raw=package_raw,
        state_raw=state_raw,
        lambda_vec=lambda_vec,
        horizon=horizon,
    )

    y = tmp[
        f"RV_avg_h{horizon}"
    ]

    X = sm.add_constant(
        tmp[
            FEATURE_COLS
        ],
        has_constant="add",
    )

    return sm.OLS(
        y,
        X,
    ).fit()


def build_origin_feature(
    package_data,
    state_data,
    lambda_vec,
):
    """
    Forecast-origin regressors at t.

    Recursive PD state is computed from all available history
    in state_data through t.

    The 1/5/22 HAR aggregation is still formed inside the formal
    package_data so the specification remains aligned with the
    existing HAR/HAR-PD rolling design.
    """
    lam_R1T, lam_RVK = np.asarray(
        lambda_vec,
        dtype=float,
    )

    combined = (
        package_data
        .copy()
        .sort_values("DT")
        .reset_index(drop=True)
    )

    combined[
        "R1T"
    ] = pd_state_tail(
        state_raw=state_data,
        package_raw=combined,
        source_col="returns",
        lam=lam_R1T,
    )

    combined[
        "RVK"
    ] = pd_state_tail(
        state_raw=state_data,
        package_raw=combined,
        source_col="RV",
        lam=lam_RVK,
    )

    combined["R1T_lag1"] = (
        combined["R1T"]
    )

    combined["R1T_lag5"] = (
        combined["R1T"]
        .rolling(5)
        .mean()
    )

    combined["R1T_lag22"] = (
        combined["R1T"]
        .rolling(22)
        .mean()
    )

    combined["RVK_lag1"] = (
        combined["RVK"]
    )

    combined["RVK_lag5"] = (
        combined["RVK"]
        .rolling(5)
        .mean()
    )

    combined["RVK_lag22"] = (
        combined["RVK"]
        .rolling(22)
        .mean()
    )

    last_row = (
        combined
        .iloc[
            [-1]
        ][
            FEATURE_COLS
        ]
        .copy()
    )

    if last_row.isna().any().any():
        raise RuntimeError(
            "Missing HAR-PD-RV regressors "
            "at forecast origin."
        )

    return sm.add_constant(
        last_row,
        has_constant="add",
    )


# ============================================================
# 9. Fixed-1500 rolling OOS | final 1000 trading days
# ============================================================
def rolling_forecast(
    data,
    horizon,
    forecast_window=FORECAST_WINDOW,
):
    data = (
        data.copy()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["DT", "RV", "returns"])
        .sort_values("DT")
        .reset_index(drop=True)
    )

    if data["DT"].duplicated().any():
        raise ValueError("Duplicate daily dates in model calendar.")

    if horizon < 1:
        raise ValueError("horizon must be >= 1.")

    n_total = len(data)

    # --------------------------------------------------------
    # Final 1000 trading days define the OOS evaluation period.
    # The formal rolling estimation package is ALWAYS 1500 days.
    #
    # First OOS target day is data[N-1000], hence its origin is
    # one trading day earlier: N-1000-1.
    # --------------------------------------------------------
    package_window = ESTIMATION_WINDOW
    first_origin = n_total - forecast_window - 1
    last_origin = n_total - horizon - 1

    expected_oos_n = forecast_window - horizon + 1
    origin_indices = np.arange(first_origin, last_origin + 1)

    first_train_start = first_origin - package_window + 1
    if first_train_start < 0:
        raise ValueError(
            f"Insufficient pre-OOS history for a fixed {package_window}-day "
            f"estimation window: N={n_total}, "
            f"forecast_window={forecast_window}."
        )

    if len(origin_indices) != expected_oos_n:
        raise RuntimeError(
            "OOS count calculation error: "
            f"expected={expected_oos_n}, actual={len(origin_indices)}."
        )

    # With the current-inclusive kernel, the first complete 22-day
    # HAR-style feature is at zero-based row 21. The last h rows do
    # not have a complete future target.
    expected_usable_n = package_window - 21 - horizon

    print("\n" + "=" * 116)
    print(
        f"HAR-PD-RV | ESTIMATION (POWELL + RAW MSE + GRID) "
        f"| FIXED-1500 | FINAL-1000 OOS | h={horizon}"
    )
    print("=" * 116)
    print(f"Total daily sample N            : {n_total}")
    print(f"Fixed raw package window       : {package_window}")
    print(f"Expected usable fit rows        : {expected_usable_n}")
    print(f"Expected OOS forecasts          : {expected_oos_n}")
    print(f"First origin                    : {data['DT'].iloc[first_origin]}")
    print(f"First forecast date             : {data['DT'].iloc[first_origin + 1]}")
    print(f"Last origin                     : {data['DT'].iloc[last_origin]}")
    print(f"Last target end                 : {data['DT'].iloc[last_origin + horizon]}")
    print("Lambda criterion                : chronological last 1% validation RAW MSE")
    print("Optimizer                       : fresh hybrid 15x15 grid + diversified top-3 Powell")
    print("Powell options                  : maxiter=500, xtol=1e-6, ftol=1e-10")
    print("Search                          : low geometric + high linear grid; 3 separated Powell starts")
    print(f"Lambda bounds                   : [{LAMBDA_LOWER}, {LAMBDA_UPPER}]")
    print("Boundary handling               : retained; diagnostic only")
    print("Warm start                      : DISABLED")
    print(f"PD state history                : expanding from {START_DATE}; OLS/validation package fixed at {ESTIMATION_WINDOW}")
    print("PD kernel timing                : CURRENT-INCLUSIVE; recursive state uses all available history")
    print("=" * 116)

    rows = []

    for forecast_number, origin_idx in enumerate(
        origin_indices,
        start=1,
    ):
        started = perf_counter()

        train_start_idx = origin_idx - package_window + 1

        # HAR-RV-aligned raw package: includes current origin t.
        package_data = (
            data.iloc[train_start_idx:origin_idx + 1]
            .copy()
            .reset_index(drop=True)
        )

        if len(package_data) != package_window:
            raise RuntimeError(
                f"Rolling package length error at forecast {forecast_number}: "
                f"expected={package_window}, actual={len(package_data)}."
            )

        # ----------------------------------------------------
        # Recursive PD-state history:
        # use ALL available observations from the configured
        # sample start through the current origin t.
        #
        # Only the trailing ESTIMATION_WINDOW observations are used
        # for validation / final OLS. Earlier observations affect
        # PD state initialization/history only.
        # ----------------------------------------------------
        state_data = (
            data.iloc[
                :origin_idx + 1
            ]
            .copy()
            .reset_index(drop=True)
        )

        prehistory_n = (
            len(state_data)
            -
            len(package_data)
        )

        if prehistory_n != train_start_idx:
            raise RuntimeError(
                "Unexpected PD-state prehistory length."
            )

        if not (
            state_data["DT"]
            .iloc[-package_window:]
            .reset_index(drop=True)
            .equals(
                package_data["DT"]
            )
        ):
            raise RuntimeError(
                "Formal package is not the trailing block of state_data."
            )

        origin_date = data["DT"].iloc[origin_idx]

        if package_data["DT"].iloc[-1] != origin_date:
            raise RuntimeError(
                "Raw package must end at forecast origin t."
            )

        actual_window = data.iloc[
            origin_idx + 1:origin_idx + horizon + 1
        ].copy()

        if len(actual_window) != horizon:
            raise RuntimeError(
                f"h={horizon}, forecast={forecast_number}: actual window has "
                f"{len(actual_window)} rows, expected {horizon}."
            )

        actual = float(actual_window["RV"].mean())

        # ----------------------------------------------------
        # STATE-HISTORY + HYBRID-GRID + DIVERSE-TOP3 POWELL; no warm start.
        # ----------------------------------------------------
        lambda_hat, selected_objective, diag = estimate_lambda(
            package_raw=package_data,
            state_raw=state_data,
            horizon=horizon,
        )


        # ----------------------------------------------------
        # UNCHANGED final OLS specification.
        # ----------------------------------------------------
        final_train = build_features(
            package_raw=package_data,
            state_raw=state_data,
            lambda_vec=lambda_hat,
            horizon=horizon,
        )

        if len(final_train) != expected_usable_n:
            raise RuntimeError(
                "Unexpected usable training rows: "
                f"expected={expected_usable_n}, actual={len(final_train)}."
            )

        model = fit_model(
            package_raw=package_data,
            state_raw=state_data,
            lambda_vec=lambda_hat,
            horizon=horizon,
        )

        X_origin = build_origin_feature(
            package_data=package_data,
            state_data=state_data,
            lambda_vec=lambda_hat,
        )

        pred_raw = float(model.predict(X_origin).iloc[0])
        pred = 1e-5 if (
            not np.isfinite(pred_raw) or pred_raw < 0
        ) else pred_raw

        # Last fitted regression origin has its h-day target ending at t.
        train_last_target_DT = package_data["DT"].iloc[-1]

        # Inner chronological-validation diagnostics from the unchanged objective.
        inner_fit_last_target_DT = package_data["DT"].iloc[
            _FAST_CONTEXT.fit_rows[-1] + horizon
        ]
        validation_first_origin_DT = package_data["DT"].iloc[
            _FAST_CONTEXT.validation_rows[0]
        ]
        validation_last_target_DT = package_data["DT"].iloc[
            _FAST_CONTEXT.validation_rows[-1] + horizon
        ]

        rows.append({
            "origin_DT": origin_date,
            "forecast_start_DT": actual_window["DT"].iloc[0],
            "forecast_end_DT": actual_window["DT"].iloc[-1],
            "train_start_DT": package_data["DT"].iloc[0],
            "train_end_DT": package_data["DT"].iloc[-1],
            "estimation_window_n": len(package_data),
            "state_history_n": len(state_data),
            "prehistory_n": prehistory_n,
            "usable_train_n": len(final_train),
            "fit_start_DT": final_train["DT"].iloc[0],
            "fit_end_DT": final_train["DT"].iloc[-1],
            "train_last_target_DT": train_last_target_DT,
            "RV_actual": actual,
            "RV_pred_raw": pred_raw,
            "RV_pred": pred,
            "selected_validation_MSE": diag["selected_validation_MSE"],
            "grid_best_validation_MSE": diag["grid_best_validation_MSE"],
            "grid_second_validation_MSE": diag[
                "grid_second_validation_MSE"
            ],
            "grid_relative_best_second_gap": diag[
                "grid_relative_best_second_gap"
            ],
            "grid_best_lambda_R1T": diag["grid_best_lambda_R1T"],
            "grid_best_lambda_RVK": diag["grid_best_lambda_RVK"],
            "powell_start1_R1T": diag["start1_R1T"],
            "powell_start1_RVK": diag["start1_RVK"],
            "powell_start2_R1T": diag["start2_R1T"],
            "powell_start2_RVK": diag["start2_RVK"],
            "powell_start3_R1T": diag["start3_R1T"],
            "powell_start3_RVK": diag["start3_RVK"],
            "grid_interior_best_validation_MSE": diag[
                "grid_interior_best_validation_MSE"
            ],
            "powell_refinements_attempted": diag[
                "powell_refinements_attempted"
            ],
            "powell_refinements_valid": diag[
                "powell_refinements_valid"
            ],
            "audit_MSE_improvement": diag["audit_MSE_improvement"],
            "search_source": diag["search_source"],
            "optimization_mode": diag["mode"],
            "boundary_count": diag["boundary_count"],
            "lambda_R1T_boundary": diag["lambda_R1T_boundary"],
            "lambda_RVK_boundary": diag["lambda_RVK_boundary"],
            f"lambda_R1T_h{horizon}": float(lambda_hat[0]),
            f"lambda_RVK_h{horizon}": float(lambda_hat[1]),
            "inner_fit_n": len(_FAST_CONTEXT.fit_rows),
            "validation_n": len(_FAST_CONTEXT.validation_rows),
            "purged_n": horizon,
            "inner_fit_last_target_DT": inner_fit_last_target_DT,
            "validation_first_origin_DT": validation_first_origin_DT,
            "validation_last_target_DT": validation_last_target_DT,
        })

        if (
            forecast_number == 1
            or forecast_number == expected_oos_n
            or forecast_number % 50 == 0
        ):
            print(
                f"HAR-PD-RV | h={horizon} | "
                f"{forecast_number}/{expected_oos_n} | "
                f"origin={origin_date.date()} | "
                f"lambda_R1T={lambda_hat[0]:.6f} | "
                f"lambda_RVK={lambda_hat[1]:.6f} | "
                f"validation_MSE={selected_objective:.10e} | "
                f"source={diag['search_source']} | "
                f"mode={diag['mode']} | "
                f"seconds={perf_counter() - started:.2f}",
                flush=True,
            )

    results = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Final timing/count audits.
    # --------------------------------------------------------
    if len(results) != expected_oos_n:
        raise RuntimeError(
            f"Final OOS count error: expected={expected_oos_n}, "
            f"actual={len(results)}."
        )

    if not (results["estimation_window_n"] == package_window).all():
        raise RuntimeError("Raw package window is not fixed.")

    if not (results["train_end_DT"] == results["origin_DT"]).all():
        raise RuntimeError("train_end_DT must equal origin_DT.")

    if not (results["train_last_target_DT"] == results["origin_DT"]).all():
        raise RuntimeError("Final training target must end at origin_DT.")

    if not (results["forecast_start_DT"] > results["origin_DT"]).all():
        raise RuntimeError("OOS target must start strictly after origin_DT.")

    actuals = results["RV_actual"].to_numpy(dtype=float)
    preds = results["RV_pred"].to_numpy(dtype=float)

    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)

    valid = (
        np.isfinite(actuals)
        & np.isfinite(preds)
        & (actuals > 0)
        & (preds > 0)
    )

    qlike = (
        float(np.mean(np.log(preds[valid]) + actuals[valid] / preds[valid]))
        if np.any(valid)
        else np.nan
    )

    print("\n" + "=" * 108)
    print(f"HAR-PD-RV | OOS RESULTS | h={horizon}")
    print("=" * 108)
    print(f"Fixed package window          : {package_window}")
    print(f"OOS observations              : {len(results)}")
    print(f"OOS origin start              : {results['origin_DT'].iloc[0]}")
    print(f"OOS origin end                : {results['origin_DT'].iloc[-1]}")
    print("Optimizer                     : hybrid 15x15 grid + diversified top-3 Powell")
    print("Lambda criterion              : chronological last 1% validation RAW MSE")
    print("Search                        : 15 x 15 grid + Powell from top 3 grid points")
    print("Boundary handling             : disabled; diagnostic only")
    print("Warm start                    : DISABLED")
    print(
        f"Boundary origins              : "
        f"{int((results['boundary_count'] > 0).sum())} / {len(results)}"
    )
    print("\n---------- OOS Loss ----------")
    print(f"MSE   : {mse:.12e}")
    print(f"RMSE  : {rmse:.12e}")
    print(f"MAE   : {mae:.12e}")
    print(f"R2    : {r2:.8f}")
    print(f"QLIKE : {qlike:.12e}")
    print("\n---------- Mean selected lambda ----------")
    print(f"lambda_R1T mean : {results[f'lambda_R1T_h{horizon}'].mean():.8f}")
    print(f"lambda_RVK mean : {results[f'lambda_RVK_h{horizon}'].mean():.8f}")
    print("=" * 108)

    return results, mse, qlike


# ============================================================
# 10. Load data and run H1 / H5 / H22
# ============================================================
def load_data():
    """
    Construct the HAR-PD-RV daily model calendar for the SSE Index.

    RV:
        computed from 5-minute intraday log returns in data_idx.csv.

    R1T return source:
        daily simple return from day_data.csv:
            returns_t = close_t / close_(t-1) - 1

    Both files are filtered to:
        code = 000001.XSHG
        date >= START_DATE
    """

    # --------------------------------------------------------
    # A. 5-minute data -> daily RV
    # --------------------------------------------------------
    data_idx = pd.read_csv(
        INTRADAY_FILE,
        low_memory=False,
    )

    required_intraday = {
        "time",
        "code",
        "close",
    }

    missing = (
        required_intraday
        - set(data_idx.columns)
    )

    if missing:
        raise ValueError(
            f"Intraday file is missing required columns: "
            f"{sorted(missing)}"
        )

    data_idx["code"] = (
        data_idx["code"]
        .astype(str)
        .str.strip()
    )

    data_idx = data_idx[
        data_idx["code"] == CODE
    ].copy()

    if data_idx.empty:
        raise ValueError(
            f"No intraday observations for {CODE}."
        )

    data_idx["datetime"] = parse_datetime(
        data_idx["time"]
    )

    data_idx["DT"] = (
        data_idx["datetime"]
        .dt.normalize()
    )

    data_idx["close"] = pd.to_numeric(
        data_idx["close"],
        errors="coerce",
    )

    data_idx = (
        data_idx
        .dropna(
            subset=[
                "datetime",
                "DT",
                "close",
            ]
        )
        .copy()
    )

    data_idx = data_idx[
        (data_idx["DT"] >= pd.Timestamp(START_DATE))
        & np.isfinite(data_idx["close"])
        & (data_idx["close"] > 0)
    ].copy()

    data_idx = (
        data_idx
        .sort_values(
            [
                "DT",
                "datetime",
            ]
        )
        .reset_index(drop=True)
    )

    duplicate_intraday = data_idx.duplicated(
        subset=[
            "DT",
            "datetime",
        ],
        keep=False,
    )

    if duplicate_intraday.any():
        print("\nDuplicate intraday timestamps:")
        print(
            data_idx.loc[
                duplicate_intraday,
                [
                    "DT",
                    "datetime",
                    "close",
                ],
            ]
            .head(20)
            .to_string(index=False)
        )
        raise ValueError(
            "Duplicate intraday timestamps found."
        )

    RV = (
        data_idx
        .groupby(
            "DT",
            sort=True,
        )
        .apply(calculate_daily_rv)
        .reset_index()
    )

    RV.columns = [
        "DT",
        "RV",
    ]

    RV["DT"] = pd.to_datetime(
        RV["DT"],
        errors="coerce",
    ).dt.normalize()

    RV["RV"] = pd.to_numeric(
        RV["RV"],
        errors="coerce",
    )

    RV = (
        RV
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "DT",
                "RV",
            ]
        )
        .loc[
            lambda x:
            x["RV"] >= 0
        ]
        .sort_values("DT")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # B. Daily file -> simple return for R1T
    # --------------------------------------------------------
    day_data = pd.read_csv(
        DAILY_FILE,
        low_memory=False,
    )

    required_daily = {
        "time",
        "code",
        "close",
    }

    missing = (
        required_daily
        - set(day_data.columns)
    )

    if missing:
        raise ValueError(
            f"Daily file is missing required columns: "
            f"{sorted(missing)}"
        )

    day_data["code"] = (
        day_data["code"]
        .astype(str)
        .str.strip()
    )

    day_data = day_data[
        day_data["code"] == CODE
    ].copy()

    if day_data.empty:
        raise ValueError(
            f"No daily observations for {CODE}."
        )

    day_data["datetime"] = parse_datetime(
        day_data["time"]
    )

    day_data["DT"] = (
        day_data["datetime"]
        .dt.normalize()
    )

    day_data["close"] = pd.to_numeric(
        day_data["close"],
        errors="coerce",
    )

    day_data = (
        day_data
        .dropna(
            subset=[
                "DT",
                "close",
            ]
        )
        .copy()
    )

    day_data = day_data[
        (day_data["DT"] >= pd.Timestamp(START_DATE))
        & np.isfinite(day_data["close"])
        & (day_data["close"] > 0)
    ].copy()

    day_data = (
        day_data
        .sort_values("DT")
        .reset_index(drop=True)
    )

    if day_data["DT"].duplicated().any():
        raise ValueError(
            "Duplicate daily dates found."
        )

    # Match the supplied HAR-RV code.
    day_data["returns"] = (
        day_data["close"]
        / day_data["close"].shift(1)
        - 1.0
    )

    returns_df = (
        day_data.loc[
            np.isfinite(
                day_data["returns"]
            ),
            [
                "DT",
                "returns",
                "close",
            ],
        ]
        .copy()
        .rename(
            columns={
                "close": "DAILY_CLOSE",
            }
        )
    )

    # --------------------------------------------------------
    # C. Common aligned daily model calendar
    # --------------------------------------------------------
    data = pd.merge(
        RV,
        returns_df,
        on="DT",
        how="inner",
        validate="one_to_one",
    )

    data = (
        data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "DT",
                "RV",
                "returns",
            ]
        )
        .sort_values("DT")
        .reset_index(drop=True)
    )

    if data["DT"].duplicated().any():
        raise ValueError(
            "Duplicate dates after RV/return merge."
        )

    minimum_required = (
        ESTIMATION_WINDOW
        + FORECAST_WINDOW
    )

    if len(data) < minimum_required:
        raise ValueError(
            f"Need at least {minimum_required} aligned daily observations "
            f"for fixed-{ESTIMATION_WINDOW} estimation and final-"
            f"{FORECAST_WINDOW} OOS evaluation; got N={len(data)}."
        )

    print("\n" + "=" * 110)
    print(f"{MODEL_LABEL} | SSE 000001.XSHG | BASE DATA")
    print("=" * 110)
    print(f"Intraday file       : {INTRADAY_FILE}")
    print(f"Daily file          : {DAILY_FILE}")
    print(f"Code filter         : {CODE}")
    print(f"Configured start    : {START_DATE}")
    print(f"Actual start        : {data['DT'].min()}")
    print(f"Actual end          : {data['DT'].max()}")
    print(f"Daily N             : {len(data)}")
    print(f"Estimation window   : {ESTIMATION_WINDOW}")
    print(f"OOS evaluation N    : {FORECAST_WINDOW}")
    print("Daily-return source : day_data.csv")
    print("Return definition   : close_t / close_(t-1) - 1")
    print("=" * 110)

    return data


if __name__ == "__main__":
    print("\nACTUAL RUNNING FILE:")
    print(os.path.abspath(__file__))
    print(f"VERSION: {MODEL_LABEL} x_t | SSE 000001.XSHG | FIXED-1500 | FINAL-1000 OOS | STATE HISTORY + HYBRID GRID + DIVERSE TOP3 POWELL | NO WARM START | RAW MSE")

    data = load_data()

    summaries = []

    for h in HORIZONS:
        result, mse, qlike = rolling_forecast(
            data=data,
            horizon=h,
            forecast_window=FORECAST_WINDOW,
        )

        output_file = f"{FILE_PREFIX}-H{h}.csv"
        result.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"Saved: {os.path.abspath(output_file)}")

        summaries.append({"Horizon": h, "MSE": mse, "QLIKE": qlike})

    print("\n" + "=" * 90)
    print(f"{MODEL_LABEL} FINAL LOSS SUMMARY")
    print("=" * 90)
    print(pd.DataFrame(summaries).to_string(index=False))
    print("=" * 90)
    print("Expected OOS rows: H1=1000, H5=996, H22=979")
    print("Only H1/H5/H22 forecast-result CSV files were written.")