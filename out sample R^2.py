import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from scipy.stats import norm


def clark_west_test(y1, y2, y):
    """
    实现Clark-West检验，计算调整后的MSPE和检验统计量。
    """
    # 确保输入是numpy数组
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    y = np.asarray(y)

    if len(y1) != len(y2) or len(y1) != len(y):
        raise ValueError("输入数组y1, y2, 和 y的长度必须一致。")


    valid_idx = ~np.isnan(y1) & ~np.isnan(y2) & ~np.isnan(y)
    y1 = y1[valid_idx]
    y2 = y2[valid_idx]
    y = y[valid_idx]


    mspe1 = np.mean((y - y1) ** 2)  # 基准模型的MSPE
    mspe2 = np.mean((y - y2) ** 2)  # 竞争模型的MSPE


    adj_term = np.mean((y1 - y2) ** 2)


    adj_mspe_diff = mspe1 - (mspe2 - adj_term)

    cw_temp = (y - y1) ** 2 - (y - y2) ** 2 + (y1 - y2) ** 2
    cw_mean = np.mean(cw_temp)

    cw_resid = cw_temp - cw_mean


    T = len(cw_temp)
    L = int(np.floor(T ** (1 / 4)))


    model = OLS(cw_resid, np.ones(T))
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': L})
    nw_se = np.sqrt(results.cov_HC0[0, 0])

    cw_stat = cw_mean / nw_se


    cw_pval = 1 - norm.cdf(cw_stat)

    oos_r2 = 1 - (mspe2 / mspe1)


    return {
        "statistic": cw_stat,
        "p.value": cw_pval,
        "mspe1": mspe1,
        "mspe2": mspe2,
        "adj_mspe_diff": adj_mspe_diff,
        "oos_r2": oos_r2
    }


import os

os.chdir("c:/Users/lenovo/Desktop/HAR")

data = pd.read_csv("1000.csv")

y = data.iloc[:, 0]  #true
y1 = data.iloc[:, 1]  # basdmodel


y2 = data.iloc[:, 2:]  # competeing model



model_names = data.columns[2:]

#Clark-West test
for model_name in model_names:
    y2 = data[model_name]


    cw_result = clark_west_test(y1, y2, y)

    print(f"Results for Model: {model_name}")
    print("MSPE for Benchmark Model (Model 1):", cw_result["mspe1"])
    print("MSPE for Competing Model (Model 2):", cw_result["mspe2"])
    print("Adjusted MSPE Difference:", cw_result["adj_mspe_diff"])
    print("Clark-West Test Statistic:", cw_result["statistic"])
    print("p-value (one-tailed):", cw_result["p.value"])
    print("Out-of-Sample R²:", cw_result["oos_r2"])
    print("\n" + "-" * 50 + "\n")
