# 导入必要的库
import pandas as pd
import numpy as np
import os

# 如果尚未安装 arch 库，请首先安装
# !pip install arch

from arch.bootstrap import MCS

# 设置工作目录（根据您的实际路径修改）
os.chdir("c:/Users/lenovo/Desktop/HAR")

# 读取数据
data = pd.read_csv("预测1步500.csv")

# 删除缺失值
data = data.dropna()

# 提取 RV_true
RV_true = data['RV']

# 获取所有包含 'har' 的列名
har_columns = [col for col in data.columns if 'har' in col]

# 提取 HAR 模型的预测结果
har_models = data[har_columns]

# 计算 ql_results
ql_results = har_models.apply(lambda x: np.log(x) + RV_true / x, axis=0)

# 计算 mse_results
mse_results = har_models.apply(lambda x: (RV_true - x) ** 2, axis=0)

# 去除异常值（将值截断在 99% 分位数）
ql_threshold = ql_results.stack().quantile(0.99)
ql_results_capped = ql_results.clip(upper=ql_threshold)

mse_threshold = mse_results.stack().quantile(0.99)
mse_results_capped = mse_results.clip(upper=mse_threshold)

# 导入用于 MCS 的库
from arch.bootstrap import MCS

# 设置随机数种子
np.random.seed(234)

# 准备数据（列为模型，行为时间序列）
ql_results_df = ql_results_capped.copy()

# 运行 MCS 程序
mcs_ql = MCS(ql_results_df, size=0.1, reps=5000)
mcs_ql_results = mcs_ql.compute()

# 输出结果
print("QL Results MCS:")
print(mcs_ql_results.pvalues)

# 对 MSE 进行相同的处理
np.random.seed(567)
mse_results_df = mse_results_capped.copy()
mcs_mse = MCS(mse_results_df, size=0.1, reps=5000)
mcs_mse_results = mcs_mse.compute()

print("\nMSE Results MCS:")
print(mcs_mse_results.pvalues)