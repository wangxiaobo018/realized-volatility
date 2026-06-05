import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.signal import lfilter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1. 读取高频数据，计算每日 RV
# ============================================================

data_idx = pd.read_csv("c:/Users/lenovo/Desktop/HAR/data_idx.csv")
data_idx = data_idx[data_idx["code"] == "000300.XSHG"].copy()

data_idx["datetime"] = pd.to_datetime(data_idx["time"])
data_idx["date"] = data_idx["datetime"].dt.date

data_idx = (
    data_idx
    .sort_values(["date", "datetime"])
    .reset_index(drop=True)
)


def calculate_daily_rv(group):

    group = group.sort_values("datetime").copy()

    price = group["close"].values
    log_ret = np.diff(np.log(price))

    if len(log_ret) < 2:
        return pd.Series({"RV": np.nan})

    return pd.Series({
        "RV": np.sum(log_ret ** 2)
    })


RV = (
    data_idx
    .groupby("date")
    .apply(calculate_daily_rv)
    .reset_index()
)

RV.columns = ["DT", "RV"]

RV["DT"] = pd.to_datetime(RV["DT"])



RV = (
    RV[RV["DT"] >= "2010-01-04"]
    .dropna(subset=["RV"])
    .sort_values("DT")
    .reset_index(drop=True)
)

# ============================================================
# 2. 读取日频数据，计算日收益率
# ============================================================

df_day = pd.read_csv("c:/Users/lenovo/Desktop/HAR/HS300.csv")
df_day = df_day[df_day["code"] == "000300.XSHG"].copy()

df_day["DT"] = pd.to_datetime(df_day["time"])
df_day = df_day.sort_values("DT").reset_index(drop=True)

df_day["returns"] = np.log(df_day["close"]).diff()
df_day["returns"] = df_day["returns"].fillna(0.0)

returns_df = df_day[["DT", "returns"]].copy()


# ============================================================
# 3. 合并 RV 与 returns
# ============================================================

data_rv = pd.merge(
    RV,
    returns_df,
    on="DT",
    how="inner"
)

data_rv = (
    data_rv
    .dropna(subset=["RV", "returns"])
    .sort_values("DT")
    .reset_index(drop=True)
)

print("=" * 60)
print("Merged data 前5行")
print("=" * 60)
print(data_rv.head())
print("样本量:", len(data_rv))


# ============================================================
# 4. 指数衰减核：只使用 t-1 及以前信息
# ============================================================

def compute_kernel_recursive(x, lam):
    x = np.asarray(x, dtype=float)
    n = len(x)

    out = np.full(n, np.nan)

    if n <= 1:
        return out

    decay = np.exp(-lam)

    b = [lam]
    a = [1, -decay]

    raw_val = lfilter(
        b,
        a,
        np.concatenate([[0.0], x[:-1]])
    )

    raw_w = lfilter(
        b,
        a,
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
# 6. 构造特征：R1T 核 + RVK 核
# ============================================================

FEATURE_COLS = [
    "R1T_lag1",
    "R1T_lag5",
    "R1T_lag22",
    "RVK_lag1",
    "RVK_lag5",
    "RVK_lag22",
]


def build_features(data_raw, lam_R1T, lam_RVK, horizon=5):
    tmp = data_raw.copy().reset_index(drop=True)

    tmp["R1T"] = compute_kernel_recursive(
        tmp["returns"].values,
        lam_R1T
    )

    tmp["RVK"] = compute_kernel_recursive(
        tmp["RV"].values,
        lam_RVK
    )

    tmp["R1T_lag1"] = tmp["R1T"]
    tmp["R1T_lag5"] = tmp["R1T"].rolling(5).mean()
    tmp["R1T_lag22"] = tmp["R1T"].rolling(22).mean()

    tmp["RVK_lag1"] = tmp["RVK"]
    tmp["RVK_lag5"] = tmp["RVK"].rolling(5).mean()
    tmp["RVK_lag22"] = tmp["RVK"].rolling(22).mean()

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

def objective_oos_mse(params, train_raw, horizon, val_frac=0.01):
    lam_R1T = float(params[0])
    lam_RVK = float(params[1])

    if lam_R1T <= 0 or lam_RVK <= 0:
        return 1e30

    if lam_R1T > 5 or lam_RVK > 5:
        return 1e30

    try:
        tmp = build_features(
            data_raw=train_raw,
            lam_R1T=lam_R1T,
            lam_RVK=lam_RVK,
            horizon=horizon
        )

        if len(tmp) < 80:
            return 1e30

        split = int(len(tmp) * (1 - val_frac))

        fit_data = tmp.iloc[:split].copy()
        val_data = tmp.iloc[split:].copy()

        if len(fit_data) < 40 or len(val_data) < 10:
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
# 8. Powell 方法估计两个 lambda
# ============================================================

def estimate_lambda(train_raw, horizon, prev_lambda=None):
    bounds = [
        (1e-4, 10.0),
        (1e-4, 10.0)
    ]

    if prev_lambda is None:
        x0 = np.array([0.10, 0.10])
    else:
        x0 = np.clip(
            np.asarray(prev_lambda, dtype=float),
            1e-4,
            5.0
        )

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

    if not opt.success:
        return float(x0[0]), float(x0[1])

    best_x = np.clip(opt.x, 1e-4, 10.0)

    return float(best_x[0]), float(best_x[1])


# ============================================================
# 9. 给定 lambda 后拟合模型
# ============================================================

def fit_model(train_raw, lam_R1T, lam_RVK, horizon):
    tmp = build_features(
        data_raw=train_raw,
        lam_R1T=lam_R1T,
        lam_RVK=lam_RVK,
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

def build_test_feature(train_raw, test_raw, lam_R1T, lam_RVK):
    combined = pd.concat(
        [train_raw, test_raw],
        axis=0
    ).reset_index(drop=True)

    combined["R1T"] = compute_kernel_recursive(
        combined["returns"].values,
        lam_R1T
    )

    combined["RVK"] = compute_kernel_recursive(
        combined["RV"].values,
        lam_RVK
    )

    combined["R1T_lag1"] = combined["R1T"]
    combined["R1T_lag5"] = combined["R1T"].rolling(5).mean()
    combined["R1T_lag22"] = combined["R1T"].rolling(22).mean()

    combined["RVK_lag1"] = combined["RVK"]
    combined["RVK_lag5"] = combined["RVK"].rolling(5).mean()
    combined["RVK_lag22"] = combined["RVK"].rolling(22).mean()

    last_row = combined.iloc[[-1]][FEATURE_COLS]

    X_test = sm.add_constant(
        last_row,
        has_constant="add"
    )

    return X_test


# ============================================================
# 11. Rolling Forecast：未来 h 日平均 RV
# ============================================================

def rolling_forecast_harpd_r1t_rvk(
    data_rv,
    horizon=5,
    forecast_window=1000
):
    data_rv = data_rv.copy().dropna().reset_index(drop=True)

    target_data = add_future_average_target(
        data_raw=data_rv,
        horizon=horizon
    )

    train_window = len(target_data) - forecast_window

    if train_window < 100:
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
    lambda_RVK_list = []

    pred_raw_list = []

    prev_lambda = None

    for i in range(forecast_window):

        train_raw = data_rv.iloc[
            i:i + train_window
        ].copy()

        origin_idx = i + train_window

        test_raw = data_rv.iloc[
            origin_idx:origin_idx + 1
        ].copy()

        target_row = target_data.iloc[
            origin_idx:origin_idx + 1
        ].copy()

        if len(test_raw) < 1 or len(target_row) < 1:
            continue

        actual = float(
            target_row[f"RV_avg_h{horizon}"].iloc[0]
        )

        forecast_end_date = target_row[f"forecast_end_DT_h{horizon}"].iloc[0]

        lam_R1T, lam_RVK = estimate_lambda(
            train_raw=train_raw,
            horizon=horizon,
            prev_lambda=prev_lambda
        )

        prev_lambda = np.array([lam_R1T, lam_RVK])

        model = fit_model(
            train_raw=train_raw,
            lam_R1T=lam_R1T,
            lam_RVK=lam_RVK,
            horizon=horizon
        )

        X_test = build_test_feature(
            train_raw=train_raw,
            test_raw=test_raw,
            lam_R1T=lam_R1T,
            lam_RVK=lam_RVK
        )

        pred_raw = float(
            model.predict(X_test).iloc[0]
        )

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
        lambda_RVK_list.append(lam_RVK)

        if (i + 1) % 50 == 0:
            print(
                f"HAR-PD-R1T-RVK future-average h={horizon} | "
                f"{i + 1}/{forecast_window} | "
                f"lambda_R1T={lam_R1T:.6f} | "
                f"lambda_RVK={lam_RVK:.6f}"
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
    print(f"HAR-PD-R1T-RVK Future-Average Rolling Forecast | h={horizon}")
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
        f"lambda_RVK_h{horizon}": lambda_RVK_list,
    })

    return results


# ============================================================
# 12. 运行 h=22
# ============================================================

FORECAST_WINDOW = 300

result_h1 = rolling_forecast_harpd_r1t_rvk(
    data_rv=data_rv,
    horizon=1,
    forecast_window=FORECAST_WINDOW
)

result_h5 = rolling_forecast_harpd_r1t_rvk(
    data_rv=data_rv,
    horizon=5,
    forecast_window=FORECAST_WINDOW
)

result_h22 = rolling_forecast_harpd_r1t_rvk(
    data_rv=data_rv,
    horizon=22,
    forecast_window=FORECAST_WINDOW
)

result_h1.to_csv("HAR-PD-RV-H1.csv", index=False)
result_h5.to_csv("HAR-PD-RV-H5.csv", index=False)
result_h22.to_csv("HAR-PD-RV-H22.csv", index=False)