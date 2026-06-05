import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 0. 基本设置
# ============================================================
INDEX_CODE = "000300.XSHG"
FORECAST_WINDOW = 1000

# ============================================================
# 1. 读取高频数据，计算每日 RV、RS+、RS-
# ============================================================

data_idx = pd.read_csv(
    "c:/Users/lenovo/Desktop/HAR/data_idx.csv"
)

data_idx = data_idx[
    data_idx["code"] == INDEX_CODE
].copy()

data_idx["datetime"] = pd.to_datetime(
    data_idx["time"]
)

# 直接筛选 2010-01-04 以后
data_idx = data_idx[
    data_idx["datetime"] >= "2010-01-04"
].copy()

data_idx["date"] = (
    data_idx["datetime"].dt.date
)

data_idx = (
    data_idx
    .sort_values(["date", "datetime"])
    .reset_index(drop=True)
)

def calculate_daily_rv_rs(group):

    group = group.sort_values(
        "datetime"
    ).copy()

    log_price = np.log(
        group["close"].values
    )

    log_ret = np.diff(log_price)

    if len(log_ret) < 2:

        return pd.Series({
            "RV": np.nan,
            "RS_plus": np.nan,
            "RS_minus": np.nan
        })

    RV = np.sum(log_ret ** 2)

    RS_plus = np.sum(
        log_ret[log_ret > 0] ** 2
    )

    RS_minus = np.sum(
        log_ret[log_ret < 0] ** 2
    )

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

rv_rs.columns = [
    "DT",
    "RV",
    "RS_plus",
    "RS_minus"
]

rv_rs["DT"] = pd.to_datetime(
    rv_rs["DT"]
)

rv_rs = (
    rv_rs
    .dropna()
    .sort_values("DT")
    .reset_index(drop=True)
)


# ============================================================
# 2. 只保留 2010-01-04 之后的数据
# ============================================================

# 3. 构造 HAR-RS 解释变量
# ============================================================

data_rs = (
    rv_rs
    .copy()
    .sort_values("DT")
    .reset_index(drop=True)
)

data_rs["RSp_lag1"] = (
    data_rs["RS_plus"].shift(1)
)

data_rs["RSp_lag5"] = (
    data_rs["RS_plus"]
    .shift(1)
    .rolling(window=5)
    .mean()
)

data_rs["RSp_lag22"] = (
    data_rs["RS_plus"]
    .shift(1)
    .rolling(window=22)
    .mean()
)

data_rs["RSm_lag1"] = (
    data_rs["RS_minus"].shift(1)
)

data_rs["RSm_lag5"] = (
    data_rs["RS_minus"]
    .shift(1)
    .rolling(window=5)
    .mean()
)

data_rs["RSm_lag22"] = (
    data_rs["RS_minus"]
    .shift(1)
    .rolling(window=22)
    .mean()
)

feature_cols = [
    "RSp_lag1",
    "RSp_lag5",
    "RSp_lag22",
    "RSm_lag1",
    "RSm_lag5",
    "RSm_lag22"
]

model_data_rs = (
    data_rs[
        ["DT", "RV"] + feature_cols
    ]
    .dropna()
    .reset_index(drop=True)
)

print("=" * 60)
print("HAR-RS model_data 前5行")
print("=" * 60)
print(model_data_rs.head())
print("model_data 样本量:", len(model_data_rs))


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
            i + 1:
            i + horizon + 1
        ]

        if len(future_window) == horizon:
            target[i] = future_window.mean()

    data["RV_h"] = target

    data["DT_h"] = (
        data["DT"]
        .shift(-horizon)
    )

    data = (
        data
        .dropna(subset=["RV_h", "DT_h"])
        .reset_index(drop=True)
    )

    return data


# ============================================================
# 5. HAR-RS 滚动预测函数
# ============================================================

def rolling_har_rs_forecast(
    model_data_rs,
    horizon=1,
    forecast_window=1000
):

    data = add_forecast_target(
        data=model_data_rs,
        horizon=horizon
    )

    train_window = (
        len(data) - forecast_window
    )

    if train_window <= 0:
        raise ValueError(
            f"h={horizon} 时样本量不足，当前有效样本量={len(data)}，"
            f"需要至少 {forecast_window + 1} 条。"
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

        # RV 预测不能为负
        pred = max(pred_raw, 1e-5)

        predictions.append(pred)

        actuals.append(
            float(test_data["RV_h"].iloc[0])
        )

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
                f"HAR-RS {label} forecast："
                f"{i + 1}/{forecast_window}"
            )

    actuals_arr = np.asarray(actuals)
    preds_arr = np.asarray(predictions)

    mse = mean_squared_error(
        actuals_arr,
        preds_arr
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        actuals_arr,
        preds_arr
    )

    r2 = r2_score(
        actuals_arr,
        preds_arr
    )

    valid_idx = preds_arr > 0

    qlike = (
        np.mean(
            np.log(preds_arr[valid_idx])
            + actuals_arr[valid_idx]
            / preds_arr[valid_idx]
        )
        if np.sum(valid_idx) > 0 else np.nan
    )

    print("=" * 60)
    if horizon == 1:
        print("HAR-RS one-step RV 预测结果")
    else:
        print(
            f"HAR-RS future-average "
            f"h={horizon} 预测结果"
        )
    print("=" * 60)

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

result_rs_h1 = (
    rolling_har_rs_forecast(
        model_data_rs=model_data_rs,
        horizon=1,
        forecast_window=FORECAST_WINDOW
    )
)

result_rs_h5 = (
    rolling_har_rs_forecast(
        model_data_rs=model_data_rs,
        horizon=5,
        forecast_window=FORECAST_WINDOW
    )
)

result_rs_h22 = (
    rolling_har_rs_forecast(
        model_data_rs=model_data_rs,
        horizon=22,
        forecast_window=FORECAST_WINDOW
    )
)


# ============================================================
# 7. 保存结果
# ============================================================

result_rs_h1.to_csv(
    "HAR-RS-H1.csv",
    index=False,
    encoding="utf-8-sig"
)

result_rs_h5.to_csv(
    "HAR-RS-H5.csv",
    index=False,
    encoding="utf-8-sig"
)

result_rs_h22.to_csv(
    "HAR-RS-H22.csv",
    index=False,
    encoding="utf-8-sig"
)