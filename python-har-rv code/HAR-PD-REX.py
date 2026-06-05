import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings


warnings.filterwarnings("ignore")


# ============================================================
data_idx = pd.read_csv("c:/Users/lenovo/Desktop/HAR/data_idx.csv")
data_idx = data_idx[data_idx["code"] == "000300.XSHG"].copy()

data_idx["datetime"] = pd.to_datetime(data_idx["time"])
data_idx["DT"] = data_idx["datetime"].dt.date
data_idx = data_idx.sort_values(["DT", "datetime"]).reset_index(drop=True)


def calculate_daily_rex(group, alpha=0.05):
    group = group.sort_values("datetime").copy()

    price = group["close"].values
    ret = np.diff(np.log(price))

    if len(ret) < 5:
        return pd.Series({
            "RV": np.nan,
            "REX_minus": np.nan,
            "REX_plus": np.nan,
            "REX_moderate": np.nan
        })

    sigma = np.std(ret, ddof=1)

    if sigma == 0 or np.isnan(sigma):
        return pd.Series({
            "RV": np.nan,
            "REX_minus": np.nan,
            "REX_plus": np.nan,
            "REX_moderate": np.nan
        })

    r_minus = norm.ppf(alpha) * sigma
    r_plus = norm.ppf(1 - alpha) * sigma

    RV = np.sum(ret ** 2)
    REX_minus = np.sum(ret[ret <= r_minus] ** 2)
    REX_plus = np.sum(ret[ret >= r_plus] ** 2)
    REX_moderate = np.sum(ret[(ret > r_minus) & (ret < r_plus)] ** 2)

    return pd.Series({
        "RV": RV,
        "REX_minus": REX_minus,
        "REX_plus": REX_plus,
        "REX_moderate": REX_moderate
    })

rex_data = (
    data_idx
    .groupby("DT", sort=True)
    .apply(calculate_daily_rex, alpha=0.05)
    .reset_index()
)

rex_data["DT"] = pd.to_datetime(rex_data["DT"])

rex_data = (
    rex_data
    .query("DT >= '2010-01-04'")
    .reset_index(drop=True)
)
# ============================================================
# 2. 日频数据：构造 returns，用于 R1T
# ============================================================

df_day = pd.read_csv("c:/Users/lenovo/Desktop/HAR/HS300.csv")
df_day = df_day[df_day["code"] == "000300.XSHG"].copy()

df_day["DT"] = pd.to_datetime(df_day["time"])
df_day = df_day.sort_values("DT").reset_index(drop=True)

# 关键修改：第一期 returns 保留为 0，保证和 RV / REX 日期对齐
df_day["returns"] = np.log(df_day["close"]).diff()
df_day["returns"] = df_day["returns"].fillna(0.0)

returns_df = df_day[["DT", "returns"]].copy()


# ============================================================
# 3. 合并 REX/RV 与 returns
# ============================================================

data_rex = pd.merge(
    rex_data,
    returns_df,
    on="DT",
    how="inner"
)

data_rex = (
    data_rex
    .dropna(subset=["RV", "REX_minus", "REX_plus", "REX_moderate", "returns"])
    .sort_values("DT")
    .reset_index(drop=True)
)

data_rex["RV_check"] = (
    data_rex["REX_minus"]
    + data_rex["REX_plus"]
    + data_rex["REX_moderate"]
)

data_rex["decomp_error"] = np.abs(data_rex["RV"] - data_rex["RV_check"])

print("=" * 60)
print("HAR-PD-REX + R1T 原始数据前5行")
print("=" * 60)
print(data_rex.head())
print("最大分解误差:", data_rex["decomp_error"].max())
print("样本量:", len(data_rex))


# ============================================================
# 4. O(n) 递推计算指数衰减核
# x[t] 的核值只使用 t-1 及以前信息
# ============================================================

def compute_pd_recursive(x, lam):
    x = np.asarray(x, dtype=float)

    n = len(x)
    out = np.full(n, np.nan)

    decay = np.exp(-lam)

    raw_val = 0.0
    raw_w = 0.0

    for t in range(1, n):
        raw_val = lam * x[t - 1] + decay * raw_val
        raw_w = lam + decay * raw_w

        if raw_w > 1e-15:
            out[t] = raw_val / raw_w

    return out


# ============================================================
# 5. 构造未来 h 日平均 RV
# ============================================================

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

    tmp[f"RV_avg_h{horizon}"] = future_average_rv(
        tmp["RV"].values,
        horizon
    )

    tmp[f"forecast_end_DT_h{horizon}"] = tmp["DT"].shift(-horizon)

    tmp = (
        tmp
        .dropna(subset=[f"RV_avg_h{horizon}", f"forecast_end_DT_h{horizon}"])
        .reset_index(drop=True)
    )

    return tmp


# ============================================================
# 6. 构造特征：R1T + PDREX- + PDREXm + PDREX+
# ============================================================

FEATURE_COLS = [
    "R1T_lag1", "R1T_lag5", "R1T_lag22",
    "PDREX_minus_lag1", "PDREX_minus_lag5", "PDREX_minus_lag22",
    "PDREX_moderate_lag1", "PDREX_moderate_lag5", "PDREX_moderate_lag22",
    "PDREX_plus_lag1", "PDREX_plus_lag5", "PDREX_plus_lag22"
]


def build_features(
    data_raw,
    lam_R1T,
    lam_minus,
    lam_moderate,
    lam_plus,
    horizon=5
):
    tmp = data_raw.copy().reset_index(drop=True)

    tmp["R1T"] = compute_pd_recursive(
        tmp["returns"].values,
        lam_R1T
    )

    tmp["PDREX_minus"] = compute_pd_recursive(
        tmp["REX_minus"].values,
        lam_minus
    )

    tmp["PDREX_moderate"] = compute_pd_recursive(
        tmp["REX_moderate"].values,
        lam_moderate
    )

    tmp["PDREX_plus"] = compute_pd_recursive(
        tmp["REX_plus"].values,
        lam_plus
    )

    for prefix, src in [
        ("R1T_lag", "R1T"),
        ("PDREX_minus_lag", "PDREX_minus"),
        ("PDREX_moderate_lag", "PDREX_moderate"),
        ("PDREX_plus_lag", "PDREX_plus")
    ]:
        s = tmp[src]

        tmp[f"{prefix}1"] = s
        tmp[f"{prefix}5"] = s.rolling(5).mean()
        tmp[f"{prefix}22"] = s.rolling(22).mean()

    tmp[f"RV_avg_h{horizon}"] = future_average_rv(
        tmp["RV"].values,
        horizon
    )

    tmp = (
        tmp
        .dropna(subset=FEATURE_COLS + [f"RV_avg_h{horizon}"])
        .reset_index(drop=True)
    )

    return tmp


# ============================================================
# 7. validation-based OOS-MSE objective
# ============================================================

def objective_oos_mse(lambda_vec, train_raw, horizon, val_frac=0.01):
    lam_R1T = float(lambda_vec[0])
    lam_minus = float(lambda_vec[1])
    lam_moderate = float(lambda_vec[2])
    lam_plus = float(lambda_vec[3])

    if lam_R1T <= 0 or lam_minus <= 0 or lam_moderate <= 0 or lam_plus <= 0:
        return 1e30

    try:
        tmp = build_features(
            data_raw=train_raw,
            lam_R1T=lam_R1T,
            lam_minus=lam_minus,
            lam_moderate=lam_moderate,
            lam_plus=lam_plus,
            horizon=horizon
        )

        if len(tmp) < 80:
            return 1e30

        split = int(len(tmp) * (1 - val_frac))

        fit_data = tmp.iloc[:split].copy()
        val_data = tmp.iloc[split:].copy()

        if len(fit_data) < 50 or len(val_data) < 10:
            return 1e30

        y_fit = fit_data[f"RV_avg_h{horizon}"]

        X_fit = sm.add_constant(
            fit_data[FEATURE_COLS],
            has_constant="add"
        )

        model = sm.OLS(y_fit, X_fit).fit()

        X_val = sm.add_constant(
            val_data[FEATURE_COLS],
            has_constant="add"
        )

        pred = model.predict(X_val)
        actual = val_data[f"RV_avg_h{horizon}"].values

        return float(mean_squared_error(actual, pred))

    except Exception:
        return 1e30


# ============================================================
# 8. 使用上一期 lambda 作为初始值估计四个 lambda
# ============================================================
def estimate_lambda(train_raw, horizon, prev_lambda=None):

    bounds = [
        (1e-4, 5.0),
        (1e-4, 5.0),
        (1e-4, 5.0),
        (1e-4, 5.0)
    ]

    if prev_lambda is not None:
        x0 = np.asarray(prev_lambda, dtype=float)
    else:
        x0 = np.array([0.10, 0.10, 0.10, 0.10])

    x0 = np.clip(x0, 1e-4, 5.0)

    opt = minimize(
        objective_oos_mse,
        x0=x0,
        args=(train_raw, horizon),
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": 300,
            "xtol": 1e-4,
            "ftol": 1e-6,
            "disp": False
        }
    )

    if opt.success:
        lambda_hat = opt.x.copy()
    else:
        lambda_hat = x0.copy()

    return np.clip(lambda_hat, 1e-4, 5.0)

# ============================================================
# 9. 给定 lambda 后，使用完整训练窗口拟合最终模型
# ============================================================

def fit_model(
    train_raw,
    lam_R1T,
    lam_minus,
    lam_moderate,
    lam_plus,
    horizon
):
    tmp = build_features(
        data_raw=train_raw,
        lam_R1T=lam_R1T,
        lam_minus=lam_minus,
        lam_moderate=lam_moderate,
        lam_plus=lam_plus,
        horizon=horizon
    )

    y = tmp[f"RV_avg_h{horizon}"]

    X = sm.add_constant(
        tmp[FEATURE_COLS],
        has_constant="add"
    )

    model = sm.OLS(y, X).fit()

    return model


# ============================================================
# 10. 构造测试点特征
# ============================================================

def build_test_feature(
    train_raw,
    test_raw,
    lam_R1T,
    lam_minus,
    lam_moderate,
    lam_plus
):
    combined = pd.concat(
        [train_raw, test_raw],
        axis=0
    ).reset_index(drop=True)

    combined["R1T"] = compute_pd_recursive(
        combined["returns"].values,
        lam_R1T
    )

    combined["PDREX_minus"] = compute_pd_recursive(
        combined["REX_minus"].values,
        lam_minus
    )

    combined["PDREX_moderate"] = compute_pd_recursive(
        combined["REX_moderate"].values,
        lam_moderate
    )

    combined["PDREX_plus"] = compute_pd_recursive(
        combined["REX_plus"].values,
        lam_plus
    )

    for prefix, src in [
        ("R1T_lag", "R1T"),
        ("PDREX_minus_lag", "PDREX_minus"),
        ("PDREX_moderate_lag", "PDREX_moderate"),
        ("PDREX_plus_lag", "PDREX_plus")
    ]:
        s = combined[src]

        combined[f"{prefix}1"] = s
        combined[f"{prefix}5"] = s.rolling(5).mean()
        combined[f"{prefix}22"] = s.rolling(22).mean()

    last_row = combined.iloc[[-1]][FEATURE_COLS]

    X_test = sm.add_constant(last_row, has_constant="add")

    return X_test


# ============================================================
# 11. Rolling Forecast：未来 h 日平均 RV
# ============================================================

def rolling_forecast_harpd_rex_r1t(
    data,
    horizon=5,
    forecast_window=1000
):
    data = data.copy().dropna().reset_index(drop=True)

    target_data = add_future_average_target(
        data_raw=data,
        horizon=horizon
    )

    train_window = len(target_data) - forecast_window

    if train_window <= 100:
        raise ValueError(
            f"训练窗口太小：当前有效样本量={len(target_data)}, "
            f"horizon={horizon}, forecast_window={forecast_window}, "
            f"train_window={train_window}"
        )

    predictions = []
    actuals = []

    origin_dates = []
    forecast_dates = []

    lambda_R1T_list = []
    lambda_minus_list = []
    lambda_moderate_list = []
    lambda_plus_list = []

    pred_raw_list = []

    prev_lambda = None

    for i in range(forecast_window):

        train_raw = data.iloc[
            i:i + train_window
        ].copy()

        origin_idx = i + train_window

        test_raw = data.iloc[
            origin_idx:origin_idx + 1
        ].copy()

        target_row = target_data.iloc[
            origin_idx:origin_idx + 1
        ].copy()

        if len(test_raw) < 1 or len(target_row) < 1:
            continue

        actual = float(target_row[f"RV_avg_h{horizon}"].iloc[0])
        forecast_end_date = target_row[f"forecast_end_DT_h{horizon}"].iloc[0]

        lambda_hat = estimate_lambda(
            train_raw=train_raw,
            horizon=horizon,
            prev_lambda=prev_lambda
        )

        lam_R1T = float(lambda_hat[0])
        lam_minus = float(lambda_hat[1])
        lam_moderate = float(lambda_hat[2])
        lam_plus = float(lambda_hat[3])

        prev_lambda = lambda_hat

        model = fit_model(
            train_raw=train_raw,
            lam_R1T=lam_R1T,
            lam_minus=lam_minus,
            lam_moderate=lam_moderate,
            lam_plus=lam_plus,
            horizon=horizon
        )

        X_test = build_test_feature(
            train_raw=train_raw,
            test_raw=test_raw,
            lam_R1T=lam_R1T,
            lam_minus=lam_minus,
            lam_moderate=lam_moderate,
            lam_plus=lam_plus
        )

        pred_raw = float(model.predict(X_test).iloc[0])

        if (not np.isfinite(pred_raw)) or (pred_raw < 0):
            pred = 1e-5
        else:
            pred = pred_raw

        pred_raw_list.append(pred_raw)
        predictions.append(pred)
        actuals.append(actual)

        origin_dates.append(test_raw["DT"].iloc[0])
        forecast_dates.append(forecast_end_date)

        lambda_R1T_list.append(lam_R1T)
        lambda_minus_list.append(lam_minus)
        lambda_moderate_list.append(lam_moderate)
        lambda_plus_list.append(lam_plus)

        if (i + 1) % 50 == 0:
            print(
                f"HAR-PD-REX-R1T future-average h={horizon} | "
                f"{i + 1}/{forecast_window} | "
                f"lambda_R1T={lam_R1T:.6f} | "
                f"lambda_minus={lam_minus:.6f} | "
                f"lambda_moderate={lam_moderate:.6f} | "
                f"lambda_plus={lam_plus:.6f}"
            )

    actuals_arr = np.asarray(actuals)
    preds_arr = np.asarray(predictions)

    mse = mean_squared_error(actuals_arr, preds_arr)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_arr, preds_arr)
    r2 = r2_score(actuals_arr, preds_arr)

    valid_idx = (
        np.isfinite(preds_arr)
        & np.isfinite(actuals_arr)
        & (preds_arr > 0)
        & (actuals_arr > 0)
    )

    if np.sum(valid_idx) == 0:
        qlike = np.nan
    else:
        qlike = np.mean(
            np.log(preds_arr[valid_idx])
            + actuals_arr[valid_idx] / preds_arr[valid_idx]
        )

    print("\n" + "=" * 60)
    print(f"HAR-PD-REX-R1T Future-Average Results | h={horizon}")
    print("=" * 60)
    print(f"MSE   : {mse:.10f}")
    print(f"RMSE  : {rmse:.10f}")
    print(f"MAE   : {mae:.10f}")
    print(f"R²    : {r2:.6f}")
    print(f"QLIKE : {qlike:.10f}")

    results = pd.DataFrame({
        "origin_DT": origin_dates,
        "forecast_end_DT": forecast_dates,
        "RV_actual": actuals,
        "RV_pred_raw": pred_raw_list,
        "RV_pred": predictions,
        f"lambda_R1T_h{horizon}": lambda_R1T_list,
        f"lambda_minus_h{horizon}": lambda_minus_list,
        f"lambda_moderate_h{horizon}": lambda_moderate_list,
        f"lambda_plus_h{horizon}": lambda_plus_list
    })

    return results


# ============================================================
# 12. 只运行 h=22
# ============================================================

FORECAST_WINDOW = 300

result_h1 = rolling_forecast_harpd_rex_r1t(
    data=data_rex,
    horizon=1,
    forecast_window=FORECAST_WINDOW
)

result_h5 = rolling_forecast_harpd_rex_r1t(
    data=data_rex,
    horizon=5,
    forecast_window=FORECAST_WINDOW
)


result_h22 = rolling_forecast_harpd_rex_r1t(
    data=data_rex,
    horizon=22,
    forecast_window=FORECAST_WINDOW
)


# ============================================================
# 13. 保存结果
# ============================================================
result_h1.to_csv(
    "HAR-PD-REX-H1.csv",
    index=False,
    encoding="utf-8-sig"
)

result_h5.to_csv(
    "HAR-PD-REX-H5.csv",
    index=False,
    encoding="utf-8-sig"
)


result_h22.to_csv(
    "HAR-PD-REX-H22.csv",
    index=False,
    encoding="utf-8-sig"
)