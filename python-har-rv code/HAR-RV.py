import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 0. 基本设置
# ============================================================


INDEX_CODE = "000300.XSHG"
FORECAST_WINDOW = 500


# ============================================================
# 1. 读取高频数据，计算每日 RV
# ============================================================


data_idx = pd.read_csv("c:/Users/lenovo/Desktop/HAR/data_idx.csv")

data_idx = data_idx[data_idx["code"] == INDEX_CODE].copy()
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

    RV = np.sum(log_ret ** 2)

    return pd.Series({"RV": RV})


data_rv = (
    data_idx
    .groupby("date", sort=True)
    .apply(calculate_daily_rv)
    .reset_index()
)

data_rv.columns = ["DT", "RV"]
data_rv["DT"] = pd.to_datetime(data_rv["DT"])

data_rv = (
    data_rv
    .dropna(subset=["RV"])
    .assign(DT=lambda x: pd.to_datetime(x["DT"]))
    .query("DT >= '2010-01-04'")
    .sort_values("DT")
    .reset_index(drop=True)
)


data_rv["RV_lag1"] = data_rv["RV"].shift(1)
data_rv["RV_lag5"] = data_rv["RV"].shift(1).rolling(window=5).mean()
data_rv["RV_lag22"] = data_rv["RV"].shift(1).rolling(window=22).mean()

model_data = (
    data_rv[["DT", "RV", "RV_lag1", "RV_lag5", "RV_lag22"]]
    .dropna()
    .reset_index(drop=True)
)

print("==============================")
print("HAR-RV model_data 前5行")
print("==============================")
print(model_data.head())
print("model_data 样本量:", len(model_data))


# ============================================================
# 4. 构造未来 h 日平均 RV
#
# h=1  : RV_{t+1}
# h=5  : mean(RV_{t+1}, ..., RV_{t+5})
# h=22 : mean(RV_{t+1}, ..., RV_{t+22})
# ============================================================

def make_future_average_rv(rv_series, horizon):
    rv = np.asarray(rv_series, dtype=float)
    n = len(rv)

    target = np.full(n, np.nan)

    for i in range(n):
        future_window = rv[i + 1:i + horizon + 1]

        if len(future_window) == horizon:
            target[i] = np.mean(future_window)

    return target


# ============================================================
# 5. 滚动预测函数：修正 h 步目标泄露
# ============================================================

def rolling_har_rv_forecast(model_data, horizon=1, forecast_window=300):

    data = (
        model_data
        .copy()
        .sort_values("DT")
        .reset_index(drop=True)
    )

    data["RV_h"] = make_future_average_rv(
        data["RV"].values,
        horizon=horizon
    )

    data["DT_h"] = data["DT"].shift(-horizon)

    data = (
        data
        .dropna(subset=["RV_h", "DT_h"])
        .reset_index(drop=True)
    )

    feature_cols = ["RV_lag1", "RV_lag5", "RV_lag22"]

    n = len(data)

    # ------------------------------------------------------------
    # 关键修正：
    # test_index 为预测原点 t
    # 训练集中最后一个样本 k 必须满足：
    # k + horizon <= test_index
    # 即训练样本的未来平均 RV 已经在预测原点前完全实现
    # ------------------------------------------------------------

    train_window = n - forecast_window - horizon + 1

    if train_window <= 50:
        raise ValueError(
            f"h={horizon} 时训练样本不足：有效样本量={n}, "
            f"forecast_window={forecast_window}, "
            f"train_window={train_window}"
        )

    predictions = []
    actuals = []
    origin_dates = []
    forecast_dates = []
    pred_raw_list = []

    for i in range(forecast_window):

        train_start = i
        train_end = i + train_window

        test_index = train_end + horizon - 1

        train_data = data.iloc[train_start:train_end].copy()
        test_data = data.iloc[test_index:test_index + 1].copy()

        if len(test_data) < 1:
            continue

        y_train = train_data["RV_h"]

        X_train = sm.add_constant(
            train_data[feature_cols],
            has_constant="add"
        )

        X_test = sm.add_constant(
            test_data[feature_cols],
            has_constant="add"
        )

        model = sm.OLS(y_train, X_train).fit()

        pred_raw = float(model.predict(X_test).iloc[0])

        pred = max(pred_raw, 1e-5)

        pred_raw_list.append(pred_raw)
        predictions.append(pred)
        actuals.append(float(test_data["RV_h"].iloc[0]))

        origin_dates.append(test_data["DT"].iloc[0])
        forecast_dates.append(test_data["DT_h"].iloc[0])

        if (i + 1) % 50 == 0:
            print(
                f"HAR-RV corrected future-average h={horizon}："
                f"已完成 {i + 1}/{forecast_window} 次滚动预测"
            )

    actuals_arr = np.asarray(actuals)
    preds_arr = np.asarray(predictions)
    pred_raw_arr = np.asarray(pred_raw_list)

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
        if np.sum(valid_idx) > 0
        else np.nan
    )

    print("==============================")
    print(f"HAR-RV corrected future-average h={horizon} 滚动预测结果")
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
        "RV_pred_raw": pred_raw_list,
        "RV_pred": predictions
    })

    return results


# ============================================================
# 6. 分别预测 h=1、h=5、h=22
# ============================================================

result_h1 = rolling_har_rv_forecast(
    model_data=model_data,
    horizon=1,
    forecast_window=FORECAST_WINDOW
)

result_h5 = rolling_har_rv_forecast(
    model_data=model_data,
    horizon=5,
    forecast_window=FORECAST_WINDOW
)

result_h22 = rolling_har_rv_forecast(
    model_data=model_data,
    horizon=22,
    forecast_window=FORECAST_WINDOW
)


# ============================================================
# 7. 保存结果
# ============================================================

result_h1.to_csv(
    "HAR-RV-H1.csv",
    index=False,
    encoding="utf-8-sig"
)

result_h5.to_csv(
    "HAR-RV-H5.csv",
    index=False,
    encoding="utf-8-sig"
)

result_h22.to_csv(
    "HAR-RV-H22.csv",
    index=False,
    encoding="utf-8-sig"
)