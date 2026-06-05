import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.signal import lfilter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1. 高频数据：构造每日 RV、RS+、RS-
# ============================================================
data_idx = pd.read_csv("c:/Users/lenovo/Desktop/HAR/data_idx.csv")
data_idx = data_idx[data_idx["code"] == "000300.XSHG"].copy()

data_idx["datetime"] = pd.to_datetime(data_idx["time"])
data_idx["date"] = data_idx["datetime"].dt.date
data_idx = data_idx.sort_values(["date", "datetime"]).reset_index(drop=True)


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
    .groupby("date")
    .apply(calculate_daily_rv_rs)
    .reset_index()
)

rv_rs.columns = ["DT", "RV", "RS_plus", "RS_minus"]
rv_rs["DT"] = pd.to_datetime(rv_rs["DT"])

data_rs = (
    rv_rs
    .dropna(subset=["RV", "RS_plus", "RS_minus"])
)

data_rs["DT"] = pd.to_datetime(data_rs["DT"])

data_rs = (
    data_rs[data_rs["DT"] >= "2010-01-04"]
    .sort_values("DT")
    .reset_index(drop=True)
)
# ============================================================
# 2. 日频数据：构造 returns，用于 R1T
# ============================================================

df_day = pd.read_csv("c:/Users/lenovo/Desktop/HAR/HS300.csv")
df_day = df_day[df_day["code"] == "000300.XSHG"].copy()

df_day["DT"] = pd.to_datetime(df_day["time"])
df_day = df_day.sort_values("DT").reset_index(drop=True)

df_day["returns"] = np.log(df_day["close"]).diff().fillna(0.0)

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

print("=" * 60)
print("RV / RS / returns 样本前5行")
print("=" * 60)
print(data.head())
print("样本量:", len(data))


# ============================================================
# 4. PD 递推函数：只使用 t-1 及以前信息
# ============================================================

def compute_pd_recursive(x, lam):
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

def make_future_rv_target(rv_series, horizon):
    rv = np.asarray(rv_series, dtype=float)
    n = len(rv)

    target = np.full(n, np.nan)

    for i in range(n):
        if i + horizon < n:
            target[i] = np.mean(rv[i + 1:i + horizon + 1])

    return target


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

    tmp["RV_target"] = make_future_rv_target(
        tmp["RV"].values,
        horizon
    )

    tmp = (
        tmp
        .dropna(subset=FEATURE_COLS + ["RV_target"])
        .reset_index(drop=True)
    )

    return tmp


# ============================================================
# 7. validation-based OOS-MSE objective
# lambda 用 OLS 估计，目标变量为原始 RV_target
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

        y_fit = fit_data["RV_target"].values
        y_val = val_data["RV_target"].values

        X_fit = sm.add_constant(
            fit_data[FEATURE_COLS],
            has_constant="add"
        )

        X_val = sm.add_constant(
            val_data[FEATURE_COLS],
            has_constant="add"
        )

        model = sm.OLS(y_fit, X_fit).fit()
        pred = model.predict(X_val)

        mse = mean_squared_error(y_val, pred)

        if np.isnan(mse) or np.isinf(mse):
            return 1e30

        return float(mse)

    except Exception:
        return 1e30


# ============================================================
# 8. 使用上一期 lambda 作为初始值估计三个 lambda
# ============================================================

def estimate_lambda(train_raw, horizon, prev_lambda=None):

    if prev_lambda is not None:
        x0 = np.asarray(prev_lambda, dtype=float)
    else:
        x0 = np.array([0.10, 0.10, 0.10])

    x0 = np.clip(x0, 1e-4, 5.0)

    opt = minimize(
        objective_oos_mse,
        x0=x0,
        args=(train_raw, horizon),
        method="Nelder-Mead",
        options={
            "xatol": 1e-5,
            "fatol": 1e-8,
            "maxiter": 500,
            "adaptive": True,
            "disp": False
        }
    )

    if opt.success and np.all(opt.x > 0):
        lambda_hat = opt.x.copy()
    else:
        lambda_hat = x0.copy()

    lambda_hat = np.clip(lambda_hat, 1e-4, 5.0)

    return float(lambda_hat[0]), float(lambda_hat[1]), float(lambda_hat[2])


# ============================================================
# 9. 固定 lambda 后，用 LassoCV 拟合最终模型
# ============================================================

def fit_model_lasso(train_raw, lam_R1T, lam_plus, lam_minus, horizon):
    tmp = build_features(
        data_raw=train_raw,
        lam_R1T=lam_R1T,
        lam_plus=lam_plus,
        lam_minus=lam_minus,
        horizon=horizon
    )

    y = tmp["RV_target"].values
    X = tmp[FEATURE_COLS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=3)

    model = LassoCV(
        cv=tscv,
        n_alphas=100,
        max_iter=10000,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_scaled, y)

    return model, scaler


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

    X_test = combined.iloc[[-1]][FEATURE_COLS]

    return X_test


# ============================================================
# 11. Rolling Forecast：未来 h 日平均 RV
# ============================================================

def rolling_forecast_lasso_harpd_rs_r1t(
    data,
    horizon=5,
    forecast_window=1000
):
    data = data.copy().dropna().reset_index(drop=True)

    train_window = len(data) - forecast_window - horizon

    if train_window <= 100:
        raise ValueError(
            f"训练窗口太小：当前样本量={len(data)}, "
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

    lasso_alpha_list = []
    coef_records = []
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

        target_raw = data.iloc[
            origin_idx + 1:origin_idx + horizon + 1
        ].copy()

        if len(target_raw) < horizon:
            continue

        actual = float(target_raw["RV"].mean())
        forecast_end_date = target_raw["DT"].iloc[-1]

        lam_R1T, lam_plus, lam_minus = estimate_lambda(
            train_raw=train_raw,
            horizon=horizon,
            prev_lambda=prev_lambda
        )

        prev_lambda = np.array([
            lam_R1T,
            lam_plus,
            lam_minus
        ])

        model, scaler = fit_model_lasso(
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

        X_test_scaled = scaler.transform(
            X_test[FEATURE_COLS]
        )

        pred_raw = float(
            model.predict(X_test_scaled)[0]
        )

        if pred_raw < 0:
            pred = 1e-5
        else:
            pred = pred_raw

        pred_raw_list.append(pred_raw)
        predictions.append(pred)
        actuals.append(actual)

        origin_dates.append(
            test_raw["DT"].iloc[0]
        )

        forecast_dates.append(
            forecast_end_date
        )

        lambda_R1T_list.append(lam_R1T)
        lambda_plus_list.append(lam_plus)
        lambda_minus_list.append(lam_minus)

        lasso_alpha_list.append(model.alpha_)
        coef_records.append(model.coef_)

        if (i + 1) % 50 == 0:
            nonzero_coef = np.sum(
                np.abs(model.coef_) > 1e-10
            )

            print(
                f"Lasso-HAR-PD-RS-R1T future-average h={horizon} | "
                f"{i + 1}/{forecast_window} | "
                f"lambda_R1T={lam_R1T:.6f} | "
                f"lambda_plus={lam_plus:.6f} | "
                f"lambda_minus={lam_minus:.6f} | "
                f"alpha={model.alpha_:.8f} | "
                f"nonzero={nonzero_coef}/{len(FEATURE_COLS)}"
            )

    actuals_arr = np.asarray(actuals)
    preds_arr = np.asarray(predictions)

    mse = mean_squared_error(actuals_arr, preds_arr)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_arr, preds_arr)
    r2 = r2_score(actuals_arr, preds_arr)

    valid_idx = (preds_arr > 0) & (actuals_arr > 0)

    qlike = (
        np.mean(
            np.log(preds_arr[valid_idx])
            + actuals_arr[valid_idx] / preds_arr[valid_idx]
        )
        if np.sum(valid_idx) > 0
        else np.nan
    )

    print("\n" + "=" * 60)
    print(f"Lasso-HAR-PD-RS-R1T Future-Average Results | h={horizon}")
    print("=" * 60)
    print(f"MSE   : {mse:.10f}")
    print(f"RMSE  : {rmse:.10f}")
    print(f"MAE   : {mae:.10f}")
    print(f"R²    : {r2:.6f}")
    print(f"QLIKE : {qlike:.10f}")

    print(f"lambda_R1T mean  : {np.mean(lambda_R1T_list):.8f}")
    print(f"lambda_plus mean : {np.mean(lambda_plus_list):.8f}")
    print(f"lambda_minus mean: {np.mean(lambda_minus_list):.8f}")
    print(f"Lasso alpha mean : {np.mean(lasso_alpha_list):.8f}")

    coef_arr = np.asarray(coef_records)

    results = pd.DataFrame({
        "origin_DT": origin_dates,
        "forecast_end_DT": forecast_dates,
        f"RV_actual_avg_h{horizon}": actuals,
        f"RV_pred_raw_avg_h{horizon}": pred_raw_list,
        f"RV_pred_avg_h{horizon}": predictions,
        f"lambda_R1T_h{horizon}": lambda_R1T_list,
        f"lambda_plus_h{horizon}": lambda_plus_list,
        f"lambda_minus_h{horizon}": lambda_minus_list,
        f"lasso_alpha_h{horizon}": lasso_alpha_list,
    })

    for j, col in enumerate(FEATURE_COLS):
        results[f"coef_{col}_h{horizon}"] = coef_arr[:, j]

    return results


# ============================================================
# 12. 运行 h=5 和 h=22
# ============================================================

FORECAST_WINDOW = 300
result_pd_r1t_rs_h1 = rolling_forecast_lasso_harpd_rs_r1t(
    data=data,
    horizon=1,
    forecast_window=FORECAST_WINDOW
)


result_pd_r1t_rs_h5 = rolling_forecast_lasso_harpd_rs_r1t(
    data=data,
    horizon=5,
    forecast_window=FORECAST_WINDOW
)

result_pd_r1t_rs_h22 = rolling_forecast_lasso_harpd_rs_r1t(
    data=data,
    horizon=22,
    forecast_window=FORECAST_WINDOW
)


# ============================================================
# 13. 保存结果
# ============================================================

result_pd_r1t_rs_h1.to_csv(
    "Lasso-HAR-PD-RS-H1.csv",
    index=False,
    encoding="utf-8-sig"
)


result_pd_r1t_rs_h5.to_csv(
    "Lasso-HAR-PD-RS-H5.csv",
    index=False,
    encoding="utf-8-sig"
)

result_pd_r1t_rs_h22.to_csv(
    "Lasso-HAR-PD-RS-H22.csv",
    index=False,
    encoding="utf-8-sig"
)