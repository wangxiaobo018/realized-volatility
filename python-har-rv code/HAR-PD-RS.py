import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.signal import lfilter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1. 读取高频数据，计算每日 RV、RS+、RS-
# ============================================================
import numpy as np
import pandas as pd

# ============================================================
# 1. 高频数据：构造每日 RV、RS+、RS-
# ============================================================

data_idx = pd.read_csv("c:/Users/lenovo/Desktop/HAR/data_idx.csv")
data_idx = data_idx[data_idx["code"] == "000300.XSHG"].copy()

data_idx["datetime"] = pd.to_datetime(data_idx["time"])
data_idx["DT"] = data_idx["datetime"].dt.date

data_idx = (
    data_idx
    .sort_values(["DT", "datetime"])
    .reset_index(drop=True)
)


def calculate_daily_rv_rs(group):
    group = group.sort_values("datetime").copy()

    log_price = np.log(group["close"].values)
    log_ret = np.diff(log_price)

    if len(log_ret) < 2:
        return pd.Series({
            "RV": np.nan,
            "RS_plus": np.nan,
            "RS_minus": np.nan
        })

    RV = np.sum(log_ret ** 2)
    RS_plus = np.sum(log_ret[log_ret > 0] ** 2)
    RS_minus = np.sum(log_ret[log_ret < 0] ** 2)

    return pd.Series({
        "RV": RV,
        "RS_plus": RS_plus,
        "RS_minus": RS_minus
    })


rv_rs = (
    data_idx
    .groupby("DT", sort=True)
    .apply(calculate_daily_rv_rs)
    .reset_index()
)

rv_rs["DT"] = pd.to_datetime(rv_rs["DT"])

data_rs = (
    rv_rs
    .dropna(subset=["RV", "RS_plus", "RS_minus"])
    .assign(DT=lambda x: pd.to_datetime(x["DT"]))
    .query("DT >= '2010-01-04'")
    .sort_values("DT")
    .reset_index(drop=True)
)

# ============================================================
# 2. 读取日频数据，计算 returns，用于 R1T
# ============================================================

df_day = pd.read_csv("c:/Users/lenovo/Desktop/HAR/HS300.csv")
df_day = df_day[df_day["code"] == "000300.XSHG"].copy()

df_day["DT"] = pd.to_datetime(df_day["time"])

df_day = (
    df_day
    .sort_values("DT")
    .reset_index(drop=True)
)

# 保留第一期 returns=0，保证和 RV / RS 日期对齐
df_day["returns"] = np.log(df_day["close"]).diff()
df_day["returns"] = df_day["returns"].fillna(0.0)

returns_df = df_day[["DT", "returns"]].copy()


# ============================================================
# 3. 合并 RV / RS / returns
# ============================================================

data = pd.merge(
    data_rs,
    returns_df,
    on="DT",
    how="inner"
)

data = (
    data
    .dropna(subset=["RV", "RS_plus", "RS_minus", "returns"])
    .sort_values("DT")
    .reset_index(drop=True)
)


# ============================================================
# 4. RS 分解检查
# ============================================================

data["RV_check"] = data["RS_plus"] + data["RS_minus"]

data["decomp_error"] = np.abs(
    data["RV"] - data["RV_check"]
)


print("=" * 60)
print("RV / RS / returns 样本前5行")
print("=" * 60)
print(data.head())
print("样本量:", len(data))


# ============================================================
# 4. lfilter 递推计算指数衰减核
# x[t] 的核值只使用 t-1 及以前的信息
# ============================================================
def compute_pd_recursive(x, lam):
    x = np.asarray(x, dtype=float)
    n = len(x)

    out = np.full(n, np.nan)

    if n == 0:
        return out

    decay = np.exp(-lam)

    raw_val = 0.0
    raw_w = 0.0

    for t in range(n):
        raw_val = lam * x[t] + decay * raw_val
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
            out[i] = np.mean(
                rv_values[i + 1:i + horizon + 1]
            )

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
# 6. 构造特征：R1T + PDRS+ + PDRS-
# ============================================================

FEATURE_COLS = [
    "R1T_lag1", "R1T_lag5", "R1T_lag22",
    "PDRS_plus_lag1", "PDRS_plus_lag5", "PDRS_plus_lag22",
    "PDRS_minus_lag1", "PDRS_minus_lag5", "PDRS_minus_lag22"
]


def build_features(data_raw, lam_R1T, lam_plus, lam_minus, horizon=5):
    tmp = data_raw.copy().reset_index(drop=True)

    tmp["R1T"] = compute_pd_recursive(
        tmp["returns"].values,
        lam_R1T
    )

    tmp["PDRS_plus"] = compute_pd_recursive(
        tmp["RS_plus"].values,
        lam_plus
    )

    tmp["PDRS_minus"] = compute_pd_recursive(
        tmp["RS_minus"].values,
        lam_minus
    )

    for prefix, src in [
        ("R1T_lag", "R1T"),
        ("PDRS_plus_lag", "PDRS_plus"),
        ("PDRS_minus_lag", "PDRS_minus")
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
    lam_plus = float(lambda_vec[1])
    lam_minus = float(lambda_vec[2])

    if lam_R1T <= 0 or lam_plus <= 0 or lam_minus <= 0:
        return 1e30

    try:
        tmp = build_features(
            data_raw=train_raw,
            lam_R1T=lam_R1T,
            lam_plus=lam_plus,
            lam_minus=lam_minus,
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
# 8. 使用上一期 lambda 作为单一初始值估计三个 lambda
# ============================================================
def estimate_lambda(train_raw, horizon, prev_lambda=None):

    bounds = [
        (1e-4, 20.0),
        (1e-4, 20.0),
        (1e-4,20.0)
    ]

    if prev_lambda is not None:
        x0 = np.asarray(prev_lambda, dtype=float)
    else:
        x0 = np.array([0.10, 0.10, 0.10])

    x0 = np.clip(x0, 1e-4, 20.0)

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

    return np.clip(lambda_hat, 1e-4, 20.0)
# ============================================================
# 9. 给定 lambda 后，用完整训练窗口拟合最终模型
# ============================================================

def fit_model(train_raw, lam_R1T, lam_plus, lam_minus, horizon):
    tmp = build_features(
        data_raw=train_raw,
        lam_R1T=lam_R1T,
        lam_plus=lam_plus,
        lam_minus=lam_minus,
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

def build_test_feature(train_raw, test_raw, lam_R1T, lam_plus, lam_minus):
    combined = pd.concat(
        [train_raw, test_raw],
        axis=0
    ).reset_index(drop=True)

    combined["R1T"] = compute_pd_recursive(
        combined["returns"].values,
        lam_R1T
    )

    combined["PDRS_plus"] = compute_pd_recursive(
        combined["RS_plus"].values,
        lam_plus
    )

    combined["PDRS_minus"] = compute_pd_recursive(
        combined["RS_minus"].values,
        lam_minus
    )

    for prefix, src in [
        ("R1T_lag", "R1T"),
        ("PDRS_plus_lag", "PDRS_plus"),
        ("PDRS_minus_lag", "PDRS_minus")
    ]:
        s = combined[src]

        combined[f"{prefix}1"] = s
        combined[f"{prefix}5"] = s.rolling(5).mean()
        combined[f"{prefix}22"] = s.rolling(22).mean()

    last_row = combined.iloc[[-1]][FEATURE_COLS]

    X_test = sm.add_constant(
        last_row,
        has_constant="add"
    )

    return X_test


# ============================================================
# 11. Rolling Forecast：未来 h 日平均 RV
# ============================================================

def rolling_forecast_harpd_r1t_rs_3lambda(
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
    lambda_plus_list = []
    lambda_minus_list = []

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

        lam_R1T, lam_plus, lam_minus = estimate_lambda(
            train_raw=train_raw,
            horizon=horizon,
            prev_lambda=prev_lambda
        )

        prev_lambda = np.array([lam_R1T, lam_plus, lam_minus])

        model = fit_model(
            train_raw=train_raw,
            lam_R1T=lam_R1T,
            lam_plus=lam_plus,
            lam_minus=lam_minus,
            horizon=horizon
        )

        X_test = build_test_feature(
            train_raw=train_raw,
            test_raw=test_raw,
            lam_R1T=lam_R1T,
            lam_plus=lam_plus,
            lam_minus=lam_minus
        )

        pred_raw = float(model.predict(X_test).iloc[0])

        # 只在预测值为负数或非有限值时替换
        if (not np.isfinite(pred_raw)) or pred_raw < 0:
            pred = 1e-5
        else:
            pred = pred_raw

        pred_raw_list.append(pred_raw)
        predictions.append(pred)
        actuals.append(actual)

        origin_dates.append(test_raw["DT"].iloc[0])
        forecast_dates.append(forecast_end_date)

        lambda_R1T_list.append(lam_R1T)
        lambda_plus_list.append(lam_plus)
        lambda_minus_list.append(lam_minus)

        if (i + 1) % 50 == 0:
            print(
                f"HAR-PD-R1T-RS-3lambda future-average h={horizon} | "
                f"{i + 1}/{forecast_window} | "
                f"lambda_R1T={lam_R1T:.6f} | "
                f"lambda_plus={lam_plus:.6f} | "
                f"lambda_minus={lam_minus:.6f}"
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
    print(f"HAR-PD-R1T-RS-3lambda Future-Average Results | h={horizon}")
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
        f"lambda_plus_h{horizon}": lambda_plus_list,
        f"lambda_minus_h{horizon}": lambda_minus_list
    })

    return results


# ============================================================
# 12. 只运行 h=22
# ============================================================

FORECAST_WINDOW = 300

result_h1 = rolling_forecast_harpd_r1t_rs_3lambda(
    data=data,
    horizon=1,
    forecast_window=FORECAST_WINDOW
)

result_h5 = rolling_forecast_harpd_r1t_rs_3lambda(
    data=data,
    horizon=5,
    forecast_window=FORECAST_WINDOW
)


result_pd_r1t_rs_h22 = rolling_forecast_harpd_r1t_rs_3lambda(
    data=data,
    horizon=22,
    forecast_window=FORECAST_WINDOW
)


# ============================================================
# 13. 保存结果
# ============================================================

result_h1.to_csv(
    "HAR-PD-RS-H1.csv",
    index=False,
    encoding="utf-8-sig"
)

result_h5.to_csv(
    "HAR-PD-RS-H5.csv",
    index=False,
    encoding="utf-8-sig"
)


result_pd_r1t_rs_h22.to_csv(
    "HAR-PD-RS-H22.csv",
    index=False,
    encoding="utf-8-sig"
)