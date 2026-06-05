import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1. 读取高频数据，计算每日 RV、REQ-、REQ+、REQm
# ============================================================
data_idx = pd.read_csv("c:/Users/lenovo/Desktop/HAR/data_idx.csv")

data_idx = data_idx[data_idx["code"] == "000300.XSHG"].copy()
data_idx["datetime"] = pd.to_datetime(data_idx["time"])
data_idx["date"] = data_idx["datetime"].dt.normalize()
data_idx = data_idx.sort_values(["date", "datetime"]).reset_index(drop=True)


def calculate_daily_req(group, alpha=0.1):

    group = group.sort_values("datetime").copy()

    log_price = np.log(group["close"].values)
    ret = np.diff(log_price)

    if len(ret) < 5:
        return pd.Series({
            "RV": np.nan,
            "REQ_minus": np.nan,
            "REQ_plus": np.nan,
            "REQ_moderate": np.nan
        })

    q_low = np.quantile(ret, alpha)
    q_high = np.quantile(ret, 1 - alpha)

    RV = np.sum(ret ** 2)

    REQ_minus = np.sum(ret[ret <= q_low] ** 2)
    REQ_plus = np.sum(ret[ret >= q_high] ** 2)
    REQ_moderate = np.sum(ret[(ret > q_low) & (ret < q_high)] ** 2)

    return pd.Series({
        "RV": RV,
        "REQ_minus": REQ_minus,
        "REQ_plus": REQ_plus,
        "REQ_moderate": REQ_moderate
    })


req_data = (
    data_idx
    .groupby("date", sort=True)
    .apply(calculate_daily_req, alpha=0.05)
    .reset_index()
)

req_data.columns = ["DT", "RV", "REQ_minus", "REQ_plus", "REQ_moderate"]

req_data["DT"] = pd.to_datetime(req_data["DT"])

req_data = (
    req_data
    .dropna()
    .sort_values("DT")
    .reset_index(drop=True)
)

req_data = (
    req_data
    .dropna()
    .assign(DT=lambda x: pd.to_datetime(x["DT"]))
    .query("DT >= '2010-01-04'")
    .sort_values("DT")
    .reset_index(drop=True)
)

data_req = req_data[[
    "DT", "RV", "REQ_minus", "REQ_plus", "REQ_moderate"
]].copy()

data_req = data_req.sort_values("DT").reset_index(drop=True)

data_req["REQm_lag1"] = data_req["REQ_moderate"].shift(1)
data_req["REQm_lag5"] = data_req["REQ_moderate"].shift(1).rolling(window=5).mean()
data_req["REQm_lag22"] = data_req["REQ_moderate"].shift(1).rolling(window=22).mean()

data_req["REQn_lag1"] = data_req["REQ_minus"].shift(1)
data_req["REQn_lag5"] = data_req["REQ_minus"].shift(1).rolling(window=5).mean()
data_req["REQn_lag22"] = data_req["REQ_minus"].shift(1).rolling(window=22).mean()

data_req["REQp_lag1"] = data_req["REQ_plus"].shift(1)
data_req["REQp_lag5"] = data_req["REQ_plus"].shift(1).rolling(window=5).mean()
data_req["REQp_lag22"] = data_req["REQ_plus"].shift(1).rolling(window=22).mean()

feature_cols = [
    "REQm_lag1", "REQm_lag5", "REQm_lag22",
    "REQn_lag1", "REQn_lag5", "REQn_lag22",
    "REQp_lag1", "REQp_lag5", "REQp_lag22"
]

model_data = (
    data_req[["DT", "RV"] + feature_cols]
    .dropna()
    .reset_index(drop=True)
)

print("==============================")
print("HAR-REQ model_data 前5行")
print("==============================")
print(model_data.head())
print("model_data 样本量:", len(model_data))


# ============================================================
# 3. 构造未来 h 日平均 RV
#
# h=5:
# RV_h(t) = mean(RV_{t+1}, ..., RV_{t+5})
#
# h=22:
# RV_h(t) = mean(RV_{t+1}, ..., RV_{t+22})
# ============================================================

def add_future_average_target(data, horizon):

    data = (
        data
        .copy()
        .sort_values("DT")
        .reset_index(drop=True)
    )

    future_avg = []

    for i in range(len(data)):

        if i + horizon >= len(data):
            future_avg.append(np.nan)
        else:
            avg_value = data["RV"].iloc[
                i + 1:i + horizon + 1
            ].mean()
            future_avg.append(avg_value)

    data["RV_h"] = future_avg
    data["DT_h"] = data["DT"].shift(-horizon)

    data = data.dropna().reset_index(drop=True)

    return data


# ============================================================
# 4. HAR-REQ 未来平均 RV 滚动预测函数
# ============================================================
def rolling_har_req_forecast_avg(
    model_data,
    horizon=5,
    forecast_window=1000
):

    data = add_future_average_target(
        data=model_data,
        horizon=horizon
    )

    # ========================================================
    # 关键修改：
    # h 步平均预测时，训练集最后一条样本的目标窗口
    # 必须在预测原点之前完全实现
    # ========================================================
    train_window = len(data) - forecast_window - horizon + 1

    if train_window <= 100:
        raise ValueError(
            f"h={horizon} 时训练窗口太小，当前有效样本量={len(data)}, "
            f"forecast_window={forecast_window}, "
            f"horizon={horizon}, train_window={train_window}"
        )

    predictions = []
    pred_raw_list = []
    actuals = []

    origin_dates = []
    forecast_dates = []

    for i in range(forecast_window):

        train_start = i
        train_end = i + train_window

        # ====================================================
        # 关键修改：
        # test_index 不能等于 train_end
        # 应该跳过 horizon - 1 个位置
        # ====================================================
        test_index = train_end + horizon - 1

        train_data = data.iloc[
            train_start:train_end
        ].copy()

        test_data = data.iloc[
            test_index:test_index + 1
        ].copy()

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

        model = sm.OLS(
            y_train,
            X_train
        ).fit()

        pred_raw = float(
            model.predict(X_test).iloc[0]
        )

        # 保存原始预测值，同时用于诊断负数
        pred = max(pred_raw, 1e-5)

        actual = float(
            test_data["RV_h"].iloc[0]
        )

        pred_raw_list.append(pred_raw)
        predictions.append(pred)
        actuals.append(actual)

        origin_dates.append(
            test_data["DT"].iloc[0]
        )

        forecast_dates.append(
            test_data["DT_h"].iloc[0]
        )

        if (i + 1) % 50 == 0:
            print(
                f"HAR-REQ corrected future-average h={horizon}："
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

    if np.sum(valid_idx) == 0:
        qlike = np.nan
    else:
        qlike = np.mean(
            np.log(preds_arr[valid_idx])
            + actuals_arr[valid_idx] / preds_arr[valid_idx]
        )

    print("==============================")
    print(f"HAR-REQ corrected 未来 {horizon} 日平均 RV 滚动预测结果")
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
# 5. 只预测未来 5 日平均 RV 和未来 22 日平均 RV
# ============================================================

forecast_window = 500


result_h1 = rolling_har_req_forecast_avg(
    model_data=model_data,
    horizon=1,
    forecast_window=forecast_window
)

result_req_h5 = rolling_har_req_forecast_avg(
    model_data=model_data,
    horizon=5,
    forecast_window=forecast_window
)

result_req_h22 = rolling_har_req_forecast_avg(
    model_data=model_data,
    horizon=22,
    forecast_window=forecast_window
)


# ============================================================
# 6. 保存结果
# ============================================================

result_h1.to_csv(
    "HAR-REQ-H1.csv",
    index=False,
    encoding="utf-8-sig"
)

result_req_h5.to_csv(
    "HAR-REQ-H5.csv",
    index=False,
    encoding="utf-8-sig"
)

result_req_h22.to_csv(
    "HAR-REQ-H22.csv",
    index=False,
    encoding="utf-8-sig"
)
