import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.stats import norm
from scipy.signal import lfilter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings


warnings.filterwarnings("ignore")


# ============================================================
# 1. 高频数据：构造每日 RV、CJ、CV
data_idx = pd.read_csv("c:/Users/lenovo/Desktop/HAR/data_idx.csv")
data_idx = data_idx[data_idx["code"] == "000300.XSHG"].copy()

data_idx["datetime"] = pd.to_datetime(data_idx["time"])
data_idx["DT"] = data_idx["datetime"].dt.date
data_idx = data_idx.sort_values(["DT", "datetime"]).reset_index(drop=True)

mu1 = np.sqrt(2 / np.pi)
mu_43 = 2 ** (2 / 3) * gamma(7 / 6) / gamma(0.5)

alpha = 0.05
q_alpha = norm.ppf(1 - alpha)


def calculate_daily_cj(group):
    group = group.sort_values("datetime").copy()

    price = group["close"].values
    ret = np.diff(np.log(price))
    n = len(ret)

    if n < 5:
        return pd.Series({
            "RV": np.nan,
            "BV": np.nan,
            "CJ": np.nan,
            "CV": np.nan,
            "Z_test": np.nan
        })

    abs_ret = np.abs(ret)
    RV = np.sum(ret ** 2)

    BV = (1 / mu1 ** 2) * np.sum(abs_ret[1:] * abs_ret[:-1])
    BV = max(BV, 0.0)

    TQ_coef = n * mu_43 ** (-3) * (n / (n - 4))

    term1 = abs_ret[2:]
    term2 = abs_ret[1:-1]
    term3 = abs_ret[:-2]

    m = min(len(term1), len(term2), len(term3))

    TQ = TQ_coef * np.sum(
        term1[:m] ** (4 / 3)
        * term2[:m] ** (4 / 3)
        * term3[:m] ** (4 / 3)
    )

    if RV <= 0 or BV <= 0 or TQ <= 0:
        return pd.Series({
            "RV": RV,
            "BV": BV,
            "CJ": 0.0,
            "CV": RV,
            "Z_test": np.nan
        })

    denom = np.sqrt(
        ((np.pi / 2) ** 2 + np.pi - 5)
        * (1 / n)
        * max(1.0, TQ / BV ** 2)
    )

    Z_test = ((RV - BV) / RV) / denom

    CJ = max(RV - BV, 0.0) if Z_test > q_alpha else 0.0
    CV = BV if Z_test > q_alpha else RV

    return pd.Series({
        "RV": RV,
        "BV": BV,
        "CJ": CJ,
        "CV": CV,
        "Z_test": Z_test
    })

data_cj = (
    data_idx
    .groupby("DT")
    .apply(calculate_daily_cj)
    .reset_index()
)

data_cj["DT"] = pd.to_datetime(data_cj["DT"])


data_cj = (
    data_cj[data_cj["DT"] >= "2010-01-04"]
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

df_day["returns"] = np.log(df_day["close"]).diff()
df_day["returns"] = df_day["returns"].fillna(0.0)

returns_df = df_day[["DT", "returns"]].copy()


# ============================================================
# 3. 合并 CJ/CV/RV 与 returns
# ============================================================

data = pd.merge(
    data_cj,
    returns_df,
    on="DT",
    how="inner"
)

data = (
    data
    .dropna(subset=["RV", "CJ", "CV", "returns"])
    .sort_values("DT")
    .reset_index(drop=True)
)

print("=" * 60)
print("数据前5行")
print("=" * 60)
print(data.head())
print("样本量:", len(data))


# ============================================================
# 4. 递推核函数：只使用 t-1 及以前信息
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
#
# h=5:
# RV_avg_h5(t) = mean(RV_{t+1}, ..., RV_{t+5})
#
# h=22:
# RV_avg_h22(t) = mean(RV_{t+1}, ..., RV_{t+22})
# ============================================================

def make_future_rv_target(rv_series, horizon):
    rv = np.asarray(rv_series, dtype=float)
    n = len(rv)

    target = np.full(n, np.nan)

    for i in range(n):
        if i + horizon < n:
            target[i] = np.mean(
                rv[i + 1:i + horizon + 1]
            )

    return target


# ============================================================
# 6. 构造特征：R1T + PDCJ + PDCV
# ============================================================

FEATURE_COLS = [
    "R1T_lag1", "R1T_lag5", "R1T_lag22",
    "PDCJ_lag1", "PDCJ_lag5", "PDCJ_lag22",
    "PDCV_lag1", "PDCV_lag5", "PDCV_lag22"
]


def build_features(data_raw, lam_R1T, lam_CJ, lam_CV, horizon=5):
    tmp = data_raw.copy().reset_index(drop=True)

    tmp["R1T"] = compute_pd_recursive(
        tmp["returns"].values,
        lam_R1T
    )

    tmp["PDCJ"] = compute_pd_recursive(
        tmp["CJ"].values,
        lam_CJ
    )

    tmp["PDCV"] = compute_pd_recursive(
        tmp["CV"].values,
        lam_CV
    )

    tmp["R1T_lag1"] = tmp["R1T"]
    tmp["R1T_lag5"] = tmp["R1T"].rolling(5).mean()
    tmp["R1T_lag22"] = tmp["R1T"].rolling(22).mean()

    tmp["PDCJ_lag1"] = tmp["PDCJ"]
    tmp["PDCJ_lag5"] = tmp["PDCJ"].rolling(5).mean()
    tmp["PDCJ_lag22"] = tmp["PDCJ"].rolling(22).mean()

    tmp["PDCV_lag1"] = tmp["PDCV"]
    tmp["PDCV_lag5"] = tmp["PDCV"].rolling(5).mean()
    tmp["PDCV_lag22"] = tmp["PDCV"].rolling(22).mean()

    tmp[f"RV_avg_h{horizon}"] = make_future_rv_target(
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
# lambda 用 OLS 估计，目标变量不取 log
# ============================================================

def objective_oos_mse(lambda_vec, train_raw, horizon, val_frac=0.2):
    lam_R1T = float(lambda_vec[0])
    lam_CJ = float(lambda_vec[1])
    lam_CV = float(lambda_vec[2])

    if lam_R1T <= 0 or lam_CJ <= 0 or lam_CV <= 0:
        return 1e30

    try:
        tmp = build_features(
            data_raw=train_raw,
            lam_R1T=lam_R1T,
            lam_CJ=lam_CJ,
            lam_CV=lam_CV,
            horizon=horizon
        )

        if len(tmp) < 80:
            return 1e30

        split = int(len(tmp) * (1 - val_frac))

        fit_data = tmp.iloc[:split].copy()
        val_data = tmp.iloc[split:].copy()

        if len(fit_data) < 50 or len(val_data) < 10:
            return 1e30

        y_fit = fit_data[f"RV_avg_h{horizon}"].values
        y_val = val_data[f"RV_avg_h{horizon}"].values

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

        return float(mean_squared_error(y_val, pred))

    except Exception:
        return 1e30


# ============================================================
# 8. 每个滚动窗口估计三个 lambda
# warm start：以上一期 lambda 作为初始值
# ============================================================

def estimate_lambda(train_raw, horizon, prev_lambda=None):
    bounds = [
        (1e-4, 10.0),
        (1e-4, 10.0),
        (1e-4, 10.0)
    ]

    if prev_lambda is None:
        x0 = np.array([0.10, 0.10, 0.10])
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
        return x0

    return np.clip(opt.x, 1e-4, 5.0)


# ============================================================
# 9. 固定 lambda 后，用 LassoCV 拟合最终模型
# ============================================================

def fit_model_lasso(train_raw, lam_R1T, lam_CJ, lam_CV, horizon):
    tmp = build_features(
        data_raw=train_raw,
        lam_R1T=lam_R1T,
        lam_CJ=lam_CJ,
        lam_CV=lam_CV,
        horizon=horizon
    )

    y = tmp[f"RV_avg_h{horizon}"].values
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

    residuals = y - model.predict(X_scaled)
    sigma2 = float(np.var(residuals, ddof=1))

    return model, scaler, sigma2


# ============================================================
# 10. 构造测试点特征
# ============================================================

def build_test_feature(train_raw, test_raw, lam_R1T, lam_CJ, lam_CV):
    combined = pd.concat(
        [train_raw, test_raw],
        axis=0
    ).reset_index(drop=True)

    combined["R1T"] = compute_pd_recursive(
        combined["returns"].values,
        lam_R1T
    )

    combined["PDCJ"] = compute_pd_recursive(
        combined["CJ"].values,
        lam_CJ
    )

    combined["PDCV"] = compute_pd_recursive(
        combined["CV"].values,
        lam_CV
    )

    combined["R1T_lag1"] = combined["R1T"]
    combined["R1T_lag5"] = combined["R1T"].rolling(5).mean()
    combined["R1T_lag22"] = combined["R1T"].rolling(22).mean()

    combined["PDCJ_lag1"] = combined["PDCJ"]
    combined["PDCJ_lag5"] = combined["PDCJ"].rolling(5).mean()
    combined["PDCJ_lag22"] = combined["PDCJ"].rolling(22).mean()

    combined["PDCV_lag1"] = combined["PDCV"]
    combined["PDCV_lag5"] = combined["PDCV"].rolling(5).mean()
    combined["PDCV_lag22"] = combined["PDCV"].rolling(22).mean()

    X_test = combined.iloc[[-1]][FEATURE_COLS]

    return X_test


# ============================================================
# 11. Rolling Forecast：未来 h 日平均 RV
# ============================================================

def rolling_forecast_lasso_harpd_cj(
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
    lambda_CJ_list = []
    lambda_CV_list = []
    lasso_alpha_list = []
    sigma2_list = []
    coef_records = []

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

        lambda_hat = estimate_lambda(
            train_raw=train_raw,
            horizon=horizon,
            prev_lambda=prev_lambda
        )

        lam_R1T = float(lambda_hat[0])
        lam_CJ = float(lambda_hat[1])
        lam_CV = float(lambda_hat[2])

        prev_lambda = lambda_hat

        model, scaler, sigma2 = fit_model_lasso(
            train_raw=train_raw,
            lam_R1T=lam_R1T,
            lam_CJ=lam_CJ,
            lam_CV=lam_CV,
            horizon=horizon
        )

        X_test = build_test_feature(
            train_raw=train_raw,
            test_raw=test_raw,
            lam_R1T=lam_R1T,
            lam_CJ=lam_CJ,
            lam_CV=lam_CV
        )

        X_test_scaled = scaler.transform(
            X_test[FEATURE_COLS]
        )

        pred_raw = float(
            model.predict(X_test_scaled)[0]
        )

        pred = pred_raw if pred_raw >= 0 else 1e-5

        predictions.append(pred)
        actuals.append(actual)

        origin_dates.append(test_raw["DT"].iloc[0])
        forecast_dates.append(forecast_end_date)

        lambda_R1T_list.append(lam_R1T)
        lambda_CJ_list.append(lam_CJ)
        lambda_CV_list.append(lam_CV)
        lasso_alpha_list.append(model.alpha_)
        sigma2_list.append(sigma2)
        coef_records.append(model.coef_)

        if (i + 1) % 50 == 0:
            nonzero_coef = np.sum(
                np.abs(model.coef_) > 1e-10
            )

            print(
                f"Lasso-HAR-PD-CJ future-average h={horizon} | "
                f"{i + 1}/{forecast_window} | "
                f"lambda_R1T={lam_R1T:.6f} | "
                f"lambda_CJ={lam_CJ:.6f} | "
                f"lambda_CV={lam_CV:.6f} | "
                f"alpha={model.alpha_:.8f} | "
                f"sigma2={sigma2:.8f} | "
                f"nonzero={nonzero_coef}/{len(FEATURE_COLS)}"
            )

    actuals_arr = np.asarray(actuals)
    preds_arr = np.asarray(predictions)

    mse = mean_squared_error(actuals_arr, preds_arr)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_arr, preds_arr)
    r2 = r2_score(actuals_arr, preds_arr)

    valid_idx = preds_arr > 0

    qlike = (
        np.mean(
            np.log(preds_arr[valid_idx])
            + actuals_arr[valid_idx] / preds_arr[valid_idx]
        )
        if np.sum(valid_idx) > 0
        else np.nan
    )

    print("\n" + "=" * 60)
    print(f"Lasso-HAR-PD-CJ Future-Average Results | h={horizon}")
    print("=" * 60)
    print(f"MSE   : {mse:.10f}")
    print(f"RMSE  : {rmse:.10f}")
    print(f"MAE   : {mae:.10f}")
    print(f"R²    : {r2:.6f}")
    print(f"QLIKE : {qlike:.10f}")

    print(f"lambda_R1T mean : {np.mean(lambda_R1T_list):.8f}")
    print(f"lambda_CJ mean  : {np.mean(lambda_CJ_list):.8f}")
    print(f"lambda_CV mean  : {np.mean(lambda_CV_list):.8f}")
    print(f"Lasso alpha mean: {np.mean(lasso_alpha_list):.8f}")
    print(f"sigma2 mean     : {np.mean(sigma2_list):.8f}")

    results = pd.DataFrame({
        "origin_DT": origin_dates,
        "forecast_end_DT": forecast_dates,
        f"RV_actual": actuals,
        f"RV_pred": predictions,
        f"lambda_R1T_h{horizon}": lambda_R1T_list,
        f"lambda_CJ_h{horizon}": lambda_CJ_list,
        f"lambda_CV_h{horizon}": lambda_CV_list,
        f"lasso_alpha_h{horizon}": lasso_alpha_list,
        f"sigma2_h{horizon}": sigma2_list,
    })

    coef_arr = np.asarray(coef_records)

    for j, col in enumerate(FEATURE_COLS):
        results[f"coef_{col}_h{horizon}"] = coef_arr[:, j]

    return results


# ============================================================
# 12. 只运行 h=5 和 h=22
# ============================================================

FORECAST_WINDOW = 300
result_h1 = rolling_forecast_lasso_harpd_cj(
    data=data,
    horizon=1,
    forecast_window=FORECAST_WINDOW
)


result_h5 = rolling_forecast_lasso_harpd_cj(
    data=data,
    horizon=5,
    forecast_window=FORECAST_WINDOW
)

result_h22 = rolling_forecast_lasso_harpd_cj(
    data=data,
    horizon=22,
    forecast_window=FORECAST_WINDOW
)


# ============================================================
# 13. 保存结果
# ============================================================
result_h1.to_csv(
    "Lasso-HAR-PD-CJ-H1.csv",
    index=False,
    encoding="utf-8-sig"
)


result_h5.to_csv(
    "Lasso-HAR-PD-CJ-H5.csv",
    index=False,
    encoding="utf-8-sig"
)

result_h22.to_csv(
    "Lasso-HAR-PD-CJ-H22.csv",
    index=False,
    encoding="utf-8-sig"
)

