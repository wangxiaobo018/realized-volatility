import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import gamma
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# 1. 读取高频数据，计算每日 RV、BV、TQ、CJ、CV
# ============================================================

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
    BV = max(BV, 0)

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
        * max(1, TQ / BV ** 2)
    )

    Z_test = ((RV - BV) / RV) / denom

    CJ = max(RV - BV, 0) if Z_test > q_alpha else 0.0
    CV = BV if Z_test > q_alpha else RV

    return pd.Series({
        "RV": RV,
        "BV": BV,
        "CJ": CJ,
        "CV": CV,
        "Z_test": Z_test
    })


data_get_cj = (
    data_idx
    .groupby("DT")
    .apply(calculate_daily_cj)
    .reset_index()
)

data_get_cj["DT"] = pd.to_datetime(data_get_cj["DT"])

data_get_cj = (
    data_get_cj
    .dropna(subset=["RV", "CJ", "CV"])
    .assign(DT=lambda x: pd.to_datetime(x["DT"]))
    .query("DT >= '2010-01-04'")
    .sort_values("DT")
    .reset_index(drop=True)
)

data_get_cj["CV_lag1"] = data_get_cj["CV"].shift(1)
data_get_cj["CV_lag5"] = data_get_cj["CV"].shift(1).rolling(window=5).mean()
data_get_cj["CV_lag22"] = data_get_cj["CV"].shift(1).rolling(window=22).mean()

data_get_cj["CJ_lag1"] = data_get_cj["CJ"].shift(1)
data_get_cj["CJ_lag5"] = data_get_cj["CJ"].shift(1).rolling(window=5).mean()
data_get_cj["CJ_lag22"] = data_get_cj["CJ"].shift(1).rolling(window=22).mean()

feature_cols = [
    "CV_lag1", "CV_lag5", "CV_lag22",
    "CJ_lag1", "CJ_lag5", "CJ_lag22"
]

model_data_cj = (
    data_get_cj[["DT", "RV"] + feature_cols]
    .dropna()
    .reset_index(drop=True)
)

print("==============================")
print("HAR-CJ model_data 前5行")
print("==============================")
print(model_data_cj.head())
print("model_data 样本量:", len(model_data_cj))


# ============================================================
# 3. 定义 HAR-CJ 多步滚动预测函数
#


def rolling_har_cj_forecast(model_data_cj, horizon=1, forecast_window=1000):

    data = model_data_cj.copy()

    data["RV_h"] = data["RV"].shift(-horizon)
    data["DT_h"] = data["DT"].shift(-horizon)

    data = data.dropna().reset_index(drop=True)

    train_window = len(data) - forecast_window

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

        train_data = data.iloc[train_start:train_end].copy()
        test_data = data.iloc[test_index:test_index + 1].copy()

        y_train = train_data["RV_h"]

        X_train = train_data[feature_cols]
        X_train = sm.add_constant(X_train, has_constant="add")

        X_test = test_data[feature_cols]
        X_test = sm.add_constant(X_test, has_constant="add")

        model = sm.OLS(y_train, X_train).fit()

        pred = model.predict(X_test).iloc[0]

        predictions.append(pred)
        actuals.append(test_data["RV_h"].iloc[0])
        origin_dates.append(test_data["DT"].iloc[0])
        forecast_dates.append(test_data["DT_h"].iloc[0])

        if (i + 1) % 50 == 0:
            print(f"HAR-CJ h={horizon}：已完成 {i + 1}/{forecast_window} 次滚动预测")

    results = pd.DataFrame({
        "origin_DT": origin_dates,
        "forecast_DT": forecast_dates,
        f"RV_actual": actuals,
        f"RV_pred": predictions
    })

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    print("==============================")
    print(f"HAR-CJ h={horizon} 步滚动预测结果")
    print("==============================")
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2 :", r2)

    return results


# ============================================================
# 4. 分别预测未来 1、5、22 步
# ============================================================

forecast_window = 1000

result_cj_h1 = rolling_har_cj_forecast(
    model_data_cj=model_data_cj,
    horizon=1,
    forecast_window=forecast_window
)

result_cj_h5 = rolling_har_cj_forecast(
    model_data_cj=model_data_cj,
    horizon=5,
    forecast_window=forecast_window
)

result_cj_h22 = rolling_har_cj_forecast(
    model_data_cj=model_data_cj,
    horizon=22,
    forecast_window=forecast_window
)


# ============================================================
# 5. 保存结果
# ============================================================

result_cj_h1.to_csv(
    "HAR-CJ-H1.csv",
    index=False,
    encoding="utf-8-sig"
)

result_cj_h5.to_csv(
    "HAR-CJ-H5.csv",
    index=False,
    encoding="utf-8-sig"
)

result_cj_h22.to_csv(
    "HAR-CJ-H22.csv",
    index=False,
    encoding="utf-8-sig"
)
