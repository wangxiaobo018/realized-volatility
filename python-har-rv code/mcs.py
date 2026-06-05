import os
import glob
import numpy as np
import pandas as pd
from arch.bootstrap import MCS

# =====================================================
# 1. 基本设置
# =====================================================
data_dir = r"D:\pycharm\doctor\新的路径依赖代码\沪深300"

#
#
# model_files = {
#     "HAR-CJ": "HAR-CJ-H1",
#     "HAR-REQ": "HAR-REQ-H1",
#     "HAR-REX": "HAR-REX-H1",
#     "HAR-RS": "HAR-RS-H1",
#     "HAR-RV": "HAR-RV-H1",
#
#    "HAR-PD-RV": "HAR-PD-RV-H1",
#    "HAR-PD-CJ": "HAR-PD-CJ-H1",
#     "HAR-PD-REX":"HAR-PD-REX-H1",
#     "HAR-PD-REQ":"HAR-PD-REQ-H1",
#     "HAR-PD-RS":"HAR-PD-RS-H1",
#     "Lasso-HAR-PD-CJ":"Lasso-HAR-PD-CJ-H1",
#     "Lasso-HAR-PD-REX":"Lasso-HAR-PD-REX-H1",
#     "Lasso-HAR-PD-REQ":"Lasso-HAR-PD-REQ-H1",
#     "Lasso-HAR-PD-RS":"Lasso-HAR-PD-RS-H1",
# }


#
#
# model_files = {
#     "HAR-CJ": "HAR-CJ-H5",
#     "HAR-REQ": "HAR-REQ-H5",
#     "HAR-REX": "HAR-REX-H5",
#     "HAR-RS": "HAR-RS-H5",
#     "HAR-RV": "HAR-RV-H5",
#
#    "HAR-PD-RV": "HAR-PD-RV-H5",
#    "HAR-PD-CJ": "HAR-PD-CJ-H5",
#     "HAR-PD-REX":"HAR-PD-REX-H5",
#     "HAR-PD-REQ":"HAR-PD-REQ-H5",
#     "HAR-PD-RS":"HAR-PD-RS-H5",
#     "Lasso-HAR-PD-CJ":"Lasso-HAR-PD-CJ-H5",
#     "Lasso-HAR-PD-REX":"Lasso-HAR-PD-REX-H5",
#     "Lasso-HAR-PD-REQ":"Lasso-HAR-PD-REQ-H5",
#     "Lasso-HAR-PD-RS":"Lasso-HAR-PD-RS-H5",
# }

#
#
model_files = {
    "HAR-CJ": "HAR-CJ-H22",
    "HAR-REQ": "HAR-REQ-H22",
    "HAR-REX": "HAR-REX-H22",
    "HAR-RS": "HAR-RS-H22",
    "HAR-RV": "HAR-RV-H22",
    "HAR-PD-RV": "HAR-PD-RV-H22",
    "HAR-PD-CJ": "HAR-PD-CJ-H22",
     "HAR-PD-REX":"HAR-PD-REX-H22",
    "HAR-PD-REQ":"HAR-PD-REQ-H22",
     "HAR-PD-RS":"HAR-PD-RS-H22",
    "Lasso-HAR-PD-CJ":"Lasso-HAR-PD-CJ-H22",
     "Lasso-HAR-PD-REX":"Lasso-HAR-PD-REX-H22",
     "Lasso-HAR-PD-REQ":"Lasso-HAR-PD-REQ-H22",
     "Lasso-HAR-PD-RS":"Lasso-HAR-PD-RS-H22",

}

eps = 1e-12


# =====================================================
# 2. 自动读取 csv / xlsx / xls
# =====================================================
def read_forecast_file(file_base):
    possible_files = []
    for ext in ["xlsx", "xls", "csv"]:
        possible_files += glob.glob(os.path.join(data_dir, file_base + "." + ext))

    if len(possible_files) == 0:
        raise FileNotFoundError(f"没有找到文件: {file_base}")

    file_path = possible_files[0]

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    df.columns = [c.strip() for c in df.columns]

    required_cols = ["RV_actual", "RV_pred"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"{file_path} 中缺少列: {c}")

    df = df[required_cols].copy()
    df["RV_actual"] = pd.to_numeric(df["RV_actual"], errors="coerce")
    df["RV_pred"] = pd.to_numeric(df["RV_pred"], errors="coerce")
    df = df.dropna()

    return df


# =====================================================
# 3. 构造损失函数
# =====================================================
loss_dict_mse = {}
loss_dict_qlike = {}

actual_ref = None

for model_name, file_base in model_files.items():
    df = read_forecast_file(file_base)

    actual = df["RV_actual"].values
    pred = df["RV_pred"].values

    pred = np.maximum(pred, eps)
    actual = np.maximum(actual, eps)

    if actual_ref is None:
        actual_ref = actual
    else:
        min_len = min(len(actual_ref), len(actual), len(pred))
        actual_ref = actual_ref[:min_len]
        actual = actual[:min_len]
        pred = pred[:min_len]

    mse_loss = (actual - pred) ** 2
    qlike_loss = np.log(pred) + actual / pred

    loss_dict_mse[model_name] = mse_loss
    loss_dict_qlike[model_name] = qlike_loss


# =====================================================
# 4. 对齐所有模型长度
# =====================================================
min_len = min(len(v) for v in loss_dict_mse.values())

loss_mse = pd.DataFrame({
    k: v[:min_len] for k, v in loss_dict_mse.items()
})

loss_qlike = pd.DataFrame({
    k: v[:min_len] for k, v in loss_dict_qlike.items()
})


# =====================================================
# 5. 输出平均损失
# =====================================================
loss_summary = pd.DataFrame({
    "MSE": loss_mse.mean(),
    "QLIKE": loss_qlike.mean()
})

loss_summary["MSE_rank"] = loss_summary["MSE"].rank()
loss_summary["QLIKE_rank"] = loss_summary["QLIKE"].rank()

print("\n================ 平均损失结果 ================")
print(loss_summary.sort_values("QLIKE"))


# =====================================================
# 6. MCS 检验函数
# =====================================================
def run_mcs(loss_df, size=0.05, reps=5000, block_size=10, method="R"):
    """
    size=0.05 表示 95% MCS
    method 可选 'R' 或 'max'
    """
    mcs = MCS(
        losses=loss_df,
        size=size,
        reps=reps,
        block_size=block_size,
        method=method,
        bootstrap="stationary"
    )

    mcs.compute()

    included = list(mcs.included)
    excluded = list(mcs.excluded)
    pvalues = mcs.pvalues

    return mcs, included, excluded, pvalues


# =====================================================
# 7. MCS 检验：MSE，size=0.05
# =====================================================
mcs_mse_95, mse_in_95, mse_out_95, mse_p_95 = run_mcs(
    loss_mse, size=0.05, reps=5000, block_size=10, method="R"
)


# =====================================================
# 8. MCS 检验：QLIKE，size=0.05
# =====================================================
mcs_qlike_95, qlike_in_95, qlike_out_95, qlike_p_95 = run_mcs(
    loss_qlike, size=0.05, reps=5000, block_size=10, method="R"
)


# =====================================================
# 9. 输出 MCS 结果
# =====================================================
print("\n================ MCS-MSE 95% ================")
print("Included models:", mse_in_95)
print("Excluded models:", mse_out_95)
print(mse_p_95)

print("\n================ MCS-QLIKE 95% ================")
print("Included models:", qlike_in_95)
print("Excluded models:", qlike_out_95)
print(qlike_p_95)