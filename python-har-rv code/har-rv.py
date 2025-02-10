import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from numba import jit
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import mstats
from scipy.optimize import minimize, differential_evolution
import statsmodels.formula.api as smf
from scipy.special import gamma
from scipy.stats import norm
from tqdm import tqdm

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import os



# get returns

def make_returns(prices):
    """
    Calculate returns from prices.
    prices: array-like of prices
    Returns: array-like of returns
    """
    n = len(prices)
    returns = np.empty(n)
    returns[0] = 0  # First return is 0

    for i in range(1, n):
        if prices[i - 1] == 0:
            returns[i] = np.nan
        else:
            returns[i] = (prices[i] - prices[i - 1]) / prices[i - 1]

    return returns

# Load data
df = pd.read_csv("c:/Users/lenovo/Desktop/HAR/day_data.csv")

# Rename columns and filter out missing values
data_ret = (
    df[['time', 'code', 'close']]  # Select relevant columns
    .rename(columns={'time': 'DT', 'code': 'id', 'close': 'PRICE'})
    .dropna()
    .groupby('id')
    .apply(lambda x: pd.DataFrame({
        'DT': x['DT'],
        'id': x['id'],
        'Ret': make_returns(x['PRICE'].values)
    }))
    .reset_index(drop=True)
)

# Group summary
group_summary = (
    data_ret.groupby('id')
    .size()
    .reset_index(name='NumObservations')
)

# Filter for a specific group ('000001.XSHG')
data_filtered = (
    df[['time', 'code', 'close']]
    .rename(columns={'time': 'DT', 'code': 'id', 'close': 'PRICE'})
    .dropna()
    .query("id == '000001.XSHG'")
)

# Calculate returns for the filtered data
# Calculate returns for the filtered data
returns = make_returns(data_filtered['PRICE'].values)

# Create a DataFrame for returns with DT from the original filtered data
returns_df = pd.DataFrame({
    'DT': data_filtered['DT'],  # Use the original 'DT' column
    'returns': returns
})

# Convert 'DT' to datetime format
returns_df['DT'] = pd.to_datetime(returns_df['DT'])


import os
# Read the data
df = pd.read_csv("c:/Users/lenovo/Desktop/HAR/data_idx.csv")

# Get group summary
group_summary = df.groupby('code').size().reset_index(name='NumObservations')

# Create data_ret DataFrame with renamed columns first
data_ret = df[['time', 'code', 'close']].copy()
data_ret.columns = ['DT', 'id', 'PRICE']
data_ret = data_ret.dropna()

# Calculate returns for each group
def calculate_returns(prices):
    returns = prices.pct_change()
    returns.iloc[0] = 0
    returns[prices.shift(1) == 0] = np.nan
    return returns

# Calculate returns by group
data_ret['Ret'] = data_ret.groupby('id')['PRICE'].transform(calculate_returns)

# Get group summary for data_ret
group_summary_ret = data_ret.groupby('id').size().reset_index(name='NumObservations')

# Filter for "000001.XSHG" and remove unnecessary columns
data_filtered = data_ret[data_ret['id'] == "000001.XSHG"].copy()
data_filtered = data_filtered.drop('id', axis=1)

# Convert DT to datetime and calculate daily RV
data_filtered['DT'] = pd.to_datetime(data_filtered['DT']).dt.date
RV = (data_filtered
      .groupby('DT')['Ret']
      .apply(lambda x: np.sum(x**2))
      .reset_index())

# Ensure RV has the correct column names
RV.columns = ['DT', 'RV']

# Convert DT to datetime for consistency with har_cj
RV['DT'] = pd.to_datetime(RV['DT'])

# Merge RV and returns_df on 'DT'
data_rv= pd.merge(RV, returns_df, on='DT', how='inner')





#Define vectorized kernel functions
def TSPL_kernel_vectorized(diff_times, lanta):
    exp_values = np.exp(-lanta * diff_times)
    return lanta * exp_values

def conv_fun1_vectorized(kernel, x, lanta):
    diff_times = np.arange(len(x))
    weights = kernel(diff_times[::-1], lanta)
    normalized_weights = weights / np.sum(weights)
    return np.sum(normalized_weights * x)

def conv_fun2_vectorized(kernel, x, lanta):
    diff_times = np.arange(len(x))
    weights = kernel(diff_times[::-1], lanta)
    normalized_weights = weights / np.sum(weights)
    return np.sum(normalized_weights * x**2)


def loss_function_vectorized(params, data, test_size=1000):
    """
    只预测一步的MSE损失函数
    """
    try:
        lanta1 = params

        # 计算特征
        r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, data['returns'][:i + 1].values, lanta1)
               for i in range(len(data))]



        # 转换为Series
        r2t = pd.Series(r2t)

        # 创建特征矩阵
        model_data = pd.DataFrame({
            'RV': data['RV'],
            'r2t_lag1': r2t.shift(1),
            'r2t_lag5': r2t.rolling(window=5).mean(),
            'r2t_lag22': r2t.rolling(window=22).mean(),

        })

        # 删除缺失值
        model_data = model_data.dropna().reset_index(drop=True)

        # 划分训练集和测试集
        train_data = model_data.iloc[:len(model_data) - test_size]
        test_data = model_data.iloc[len(model_data) - test_size:]

        # 准备训练数据
        X_train = train_data.drop('RV', axis=1)
        y_train = train_data['RV']
        X_test = test_data.drop('RV', axis=1)
        y_test = test_data['RV']

        # 初始化预测误差列表
        mse_steps = []

        # 使用滚动窗口进行预测
        rolling_X = X_train.copy()
        rolling_y = y_train.copy()

        for i in range(len(X_test)):
            # 训练模型
            model = LinearRegression()
            model.fit(rolling_X, rolling_y)

            # 一步预测
            pred = model.predict(X_test[i:i + 1])
            mse_steps.append((pred[0] - y_test.iloc[i]) ** 2)

            # 更新滚动窗口
            rolling_X = np.vstack((rolling_X[1:], X_test.iloc[i:i + 1].values))
            rolling_y = np.concatenate((rolling_y[1:], [y_test.iloc[i]]))

        # 计算MSE
        mse = np.mean(mse_steps)

        return mse

    except Exception as e:
        print(f"Error in loss function: {str(e)}")
        return np.inf


def optimize_parameters(data, test_size=800):
    # 设置优化参数
    de_result = differential_evolution(
        func=loss_function_vectorized,
        bounds=[(1e-6, 40)] ,
        args=(data, test_size),
        strategy='best1bin',
        maxiter=200,
        popsize=30,
        tol=0.01,
        disp=True,
        workers=-1  # 使用并行计算
    )

    # 输出结果
    print("\n优化结果:")
    print(f"最优参数值: lanta1={de_result.x[0]:.6f}")
    print(f"最小MSE值: {de_result.fun:.6f}")

    return de_result

# 主程序入口
if __name__ == "__main__":
    result = optimize_parameters(data_rv)

# #300 下 0.51344379
# r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, data_rv['returns'].iloc[:i+1].values, lanta1)
#        for i in range(len(data_rv))]
#
#
# r2t = pd.Series(r2t)
#
# model_data = pd.DataFrame({
#     'RV': RV['RV'],
#     'r2t_lag1': r2t.shift(1),
#     'r2t_lag5': r2t.rolling(window=5).mean(),
#     'r2t_lag22': r2t.rolling(window=22).mean()
# })
#
#
# # 删除包含缺失值的行
# model_data = model_data.dropna()
#
# # 定义因变量 y 和自变量 X
# y = model_data['RV']
# X = model_data.drop('RV', axis=1)
#
# # 检查 X 和 y 是否为空
# print(f"y is empty: {y.empty}")
# print(f"X is empty: {X.empty}")
#
# # 添加常数项
# X = sm.add_constant(X)
#
# # 确保没有零大小的数组
# if y.empty or X.empty:
#     print("Error: The data is empty or invalid.")
# else:
#     # 拟合线性回归模型
#     model = sm.OLS(y, X).fit()
#
#     # 获取并打印模型的详细统计摘要
#     summary = model.summary()
#     print(summary)
#
#
#
# test_size = 800
#
# # 划分训练集和测试集
# train_data = model_data.iloc[:len(model_data) - test_size]
# test_data = model_data.iloc[len(model_data) - test_size:]
#
# # 分割特征和目标值
# X_train = train_data.drop('RV', axis=1)
# y_train = train_data['RV']
# X_test = test_data.drop('RV', axis=1)
# y_test = test_data['RV']
#
# # Initialize prediction and actual value lists
# predictions_lr1 = []
# actuals_lr1 = []
# predictions_lr5 = []
# actuals_lr5 = []
# predictions_lr22 = []
# actuals_lr22 = []
#
# # Initialize rolling window with training data
# rolling_X = X_train.copy()
# rolling_y = y_train.copy()
#
# # Rolling window prediction
# for i in range(len(X_test)):
#     # Train model on current window
#     model = LinearRegression()
#     model.fit(rolling_X, rolling_y)
#
#     # 1-step ahead prediction (单步预测)
#     pred_1 = model.predict(X_test[i:i + 1])
#     predictions_lr1.append(pred_1[0])
#     actuals_lr1.append(y_test.iloc[i])
#
#     # 5-step ahead prediction (5步预测)
#     if i + 4 < len(X_test):
#         pred_5 = model.predict(X_test.iloc[i:i + 5])
#         predictions_lr5.append(pred_5[-1])
#         actuals_lr5.append(y_test.iloc[i + 4])
#     else:
#         predictions_lr5.append(None)
#         actuals_lr5.append(None)
#
#     # 22-step ahead prediction (22步预测)
#     if i + 21 < len(X_test):
#         pred_22 = model.predict(X_test.iloc[i:i + 22])
#         predictions_lr22.append(pred_22[-1])
#         actuals_lr22.append(y_test.iloc[i + 21])
#     else:
#         predictions_lr22.append(None)
#         actuals_lr22.append(None)
#
#     # Update rolling window by removing oldest observation and adding new one
#     rolling_X = np.vstack((rolling_X[1:], X_test.iloc[i:i + 1].values))
#     rolling_y = np.concatenate((rolling_y[1:], [y_test.iloc[i]]))
#
# # Create results DataFrame to store predictions and actuals
# df_predictions_lr = pd.DataFrame({
#     'Prediction_1': predictions_lr1,
#     'Actual_1': actuals_lr1,
#     'Prediction_5': predictions_lr5,
#     'Actual_5': actuals_lr5,
#     'Prediction_22': predictions_lr22,
#     'Actual_22': actuals_lr22
# })
#
#
# df_predictions_lr.to_csv('rv_pd800.csv', index=False)