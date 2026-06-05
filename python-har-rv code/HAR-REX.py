import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 0. 基本设置
# ============================================================


INDEX_CODE = "000300.XSHG"
FORECAST_WINDOW = 300


# ============================================================
# 1. 读取高频数据，计算每日 RV、REX-、REX+、REXm
# ============================================================
data_idx = pd.read_csv("c:/Users/lenovo/Desktop/HAR/data_idx.csv")
data_idx = data_idx[data_idx["code"] == INDEX_CODE].copy()
data_idx["datetime"] = pd.to_datetime(data_idx["time"])
data_idx["date"] = data_idx["datetime"].dt.normalize()

data_idx = (
    data_idx
    .sort_values(["date", "datetime"])
    .reset_index(drop=True)
)


def calculate_daily_rex(group, alpha=0.05):
    group = group.sort_values("datetime").copy()

    log_price = np.log(group["close"].values)
    ret = np.diff(log_price)

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
    REX_moderate = np.sum(
        ret[(ret > r_minus) & (ret < r_plus)] ** 2
    )

    return pd.Series({
        "RV": RV,
        "REX_minus": REX_minus,
        "REX_plus": REX_plus,
        "REX_moderate": REX_moderate
    })


rex_data = (
    data_idx
    .groupby("date", sort=True)
    .apply(calculate_daily_rex, alpha=0.05)
    .reset_index()
)

rex_data.columns = [
    "DT",
    "RV",
    "REX_minus",
    "REX_plus",
    "REX_moderate"
]

rex_data["DT"] = pd.to_datetime(rex_data["DT"])



rex_data = (
    rex_data[rex_data["DT"] >= "2010-01-04"]
    .dropna()
    .sort_values("DT")
    .reset_index(drop=True)
)

# ============================================================
# ============================================================
# 3. 构建 HAR-REX 模型特征
# ============================================================

data_rex = rex_data[
    [
        "DT",
        "RV",
        "REX_minus",
        "REX_plus",
        "REX_moderate"
    ]
].copy()

data_rex["DT"] = pd.to_datetime(data_rex["DT"])

data_rex = (
    data_rex[data_rex["DT"] >= "2010-01-04"]
    .sort_values("DT")
    .reset_index(drop=True)
)

data_rex = (
    data_rex
    .sort_values("DT")
    .reset_index(drop=True)
)

data_rex["REXm_lag1"] = data_rex["REX_moderate"].shift(1)
data_rex["REXm_lag5"] = (
    data_rex["REX_moderate"]
    .shift(1)
    .rolling(window=5)
    .mean()
)
data_rex["REXm_lag22"] = (
    data_rex["REX_moderate"]
    .shift(1)
    .rolling(window=22)
    .mean()
)

data_rex["REXn_lag1"] = data_rex["REX_minus"].shift(1)
data_rex["REXn_lag5"] = (
    data_rex["REX_minus"]
    .shift(1)
    .rolling(window=5)
    .mean()
)
data_rex["REXn_lag22"] = (
    data_rex["REX_minus"]
    .shift(1)
    .rolling(window=22)
    .mean()
)

data_rex["REXp_lag1"] = data_rex["REX_plus"].shift(1)
data_rex["REXp_lag5"] = (
    data_rex["REX_plus"]
    .shift(1)
    .rolling(window=5)
    .mean()
)
data_rex["REXp_lag22"] = (
    data_rex["REX_plus"]
    .shift(1)
    .rolling(window=22)
    .mean()
)

feature_cols = [
    "REXm_lag1", "REXm_lag5", "REXm_lag22",
    "REXn_lag1", "REXn_lag5", "REXn_lag22",
    "REXp_lag1", "REXp_lag5", "REXp_lag22"
]

model_data = (
    data_rex[["DT", "RV"] + feature_cols]
    .dropna()
    .reset_index(drop=True)
)

print("==============================")
print("HAR-REX model_data 前5行")
print("==============================")
print(model_data.head())
print("model_data 样本量:", len(model_data))


# ============================================================
# 4. 构造预测目标
#
# h=1  : RV(t+1)
# h=5  : mean(RV(t+1), ..., RV(t+5))
# h=22 : mean(RV(t+1), ..., RV(t+22))
# ============================================================

def add_forecast_target(data, horizon):
    data = (
        data
        .copy()
        .sort_values("DT")
        .reset_index(drop=True)
    )

    target = np.full(len(data), np.nan)

    for i in range(len(data)):
        future_window = data["RV"].iloc[
            i + 1:i + horizon + 1
        ]

        if len(future_window) == horizon:
            target[i] = future_window.mean()

    data["RV_h"] = target
    data["DT_h"] = data["DT"].shift(-horizon)

    data = (
        data
        .dropna(subset=["RV_h", "DT_h"])
        .reset_index(drop=True)
    )

    return data


# ============================================================
# 5. HAR-REX 滚动预测函数
# ============================================================

def rolling_har_rex_forecast(
    model_data,
    horizon=1,
    forecast_window=1000
):

    data = add_forecast_target(
        data=model_data,
        horizon=horizon
    )

    train_window = len(data) - forecast_window

    if train_window <= 100:
        raise ValueError(
            f"h={horizon} 时训练窗口太小，当前有效样本量={len(data)}, "
            f"forecast_window={forecast_window}, "
            f"train_window={train_window}"
        )

    predictions = []
    actuals = []

    origin_dates = []
    forecast_dates = []

    for i in range(forecast_window):

        train_start = i
        train_end = i + train_window
        test_index = train_end

        train_data = data.iloc[
            train_start:train_end
        ].copy()

        test_data = data.iloc[
            test_index:test_index + 1
        ].copy()

        y_train = train_data["RV_h"]

        X_train = sm.add_constant(
            train_data[feature_cols],
            has_constant="add"
        )

        X_test = sm.add_constant(
            test_data[feature_cols],
            has_constant="add"
        )

        model = sm.OLS(
            y_train,
            X_train
        ).fit()

        pred_raw = float(
            model.predict(X_test).iloc[0]
        )

        # RV 预测值不能为负
        if (not np.isfinite(pred_raw)) or (pred_raw < 0):
            pred = 1e-5
        else:
            pred = pred_raw

        actual = float(
            test_data["RV_h"].iloc[0]
        )

        predictions.append(pred)
        actuals.append(actual)

        origin_dates.append(
            test_data["DT"].iloc[0]
        )

        forecast_dates.append(
            test_data["DT_h"].iloc[0]
        )

        if (i + 1) % 50 == 0:

            if horizon == 1:
                label = "one-step"
            else:
                label = f"{horizon}-day future-average"

            print(
                f"HAR-REX {label} forecast："
                f"已完成 {i + 1}/{forecast_window} 次滚动预测"
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

    qlike = (
        np.mean(
            np.log(preds_arr[valid_idx])
            + actuals_arr[valid_idx] / preds_arr[valid_idx]
        )
        if np.sum(valid_idx) > 0 else np.nan
    )

    print("==============================")
    if horizon == 1:
        print("HAR-REX one-step RV 滚动预测结果")
    else:
        print(f"HAR-REX 未来 {horizon} 日平均 RV 滚动预测结果")
    print("==============================")
    print(f"MSE   : {mse:.10f}")
    print(f"RMSE  : {rmse:.10f}")
    print(f"MAE   : {mae:.10f}")
    print(f"R2    : {r2:.6f}")
    print(f"QLIKE : {qlike:.10f}")

    results = pd.DataFrame({
        "origin_DT": origin_dates,
        "forecast_end_DT": forecast_dates,
        "RV_actual": actuals,
        "RV_pred": predictions
    })

    return results


# ============================================================
# 6. 分别预测 h=1、h=5、h=22
# ============================================================

result_rex_h1 = rolling_har_rex_forecast(
    model_data=model_data,
    horizon=1,
    forecast_window=FORECAST_WINDOW
)

result_rex_h5 = rolling_har_rex_forecast(
    model_data=model_data,
    horizon=5,
    forecast_window=FORECAST_WINDOW
)

result_rex_h22 = rolling_har_rex_forecast(
    model_data=model_data,
    horizon=22,
    forecast_window=FORECAST_WINDOW
)


# ============================================================
# 7. 保存结果
# ============================================================

result_rex_h1.to_csv(
    "HAR-REX-H1.csv",
    index=False,
    encoding="utf-8-sig"
)

result_rex_h5.to_csv(
    "HAR-REX-H5.csv",
    index=False,
    encoding="utf-8-sig"
)

result_rex_h22.to_csv(
    "HAR-REX-H22.csv",
    index=False,
    encoding="utf-8-sig"
)