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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import os

# #上证指数
# os.chdir("c:/Users/lenovo/Desktop/HAR")
#
#
# df = pd.read_csv("data_idx.csv")
#
#
# data_filtered = df[df['code'] == "000001.XSHG"].copy()
#
#

# def get_RV_BV(data, alpha=0.05, times=True):
#
#     idx = 100 if times else 1
#
#     df = data.copy()
#
#
#     df['datetime'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H:%M')
#     df['day'] = df['datetime'].dt.date
#
#     results = []
#     for day, group in df.groupby('day'):
#
#         group = group.sort_values('datetime')
#
#
#         group['Ret'] = (np.log(group['close']) - np.log(group['close'].shift(1))) * idx
#         group['Ret_HL'] = (np.log(group['high']) - np.log(group['low'].shift(1))) * idx
#
#
#         group = group.dropna(subset=['Ret', 'Ret_HL'])
#         n = len(group)
#
#         if n < 5:
#             continue
#
#         # 计算RV
#         RV = np.sum(group['Ret'] ** 2)
#
#         # 计算BV
#         abs_ret = np.abs(group['Ret'])
#         BV = (np.pi / 2) * np.sum(abs_ret.shift(1) * abs_ret.shift(-1).dropna())
#
#
#         TQ_coef = n * (2 ** (2 / 3) * gamma(7 / 6) / gamma(0.5)) ** (-3) * (n / (n - 4))
#
#
#         term1 = abs_ret.iloc[4:].values  # Ret[5:n()]
#         term2 = abs_ret.iloc[2:-2].values  # Ret[3:(n-2)]
#         term3 = abs_ret.iloc[:-4].values  # Ret[1:(n-4)]
#
#         min_len = min(len(term1), len(term2), len(term3))
#         if min_len > 0:
#             TQ = TQ_coef * np.sum((term1[:min_len] ** (4 / 3)) *
#                                   (term2[:min_len] ** (4 / 3)) *
#                                   (term3[:min_len] ** (4 / 3)))
#         else:
#             continue
#
#         # Z_test
#         Z_test = ((RV - BV) / RV) / np.sqrt(((np.pi / 2) ** 2 + np.pi - 5) *
#                                             (1 / n) * max(1, TQ / (BV ** 2)))
#
#         # calculate JV
#         q_alpha = norm.ppf(1 - alpha)
#         JV = (RV - BV) * (Z_test > q_alpha)
#         C_t = (Z_test <= q_alpha) * RV + (Z_test > q_alpha) * BV
#
#         results.append({
#
#             'BV': BV,
#             'JV': JV,
#             'C_t': C_t
#         })
#
#
#     result_df = pd.DataFrame(results)
#     return result_df[['BV', 'JV', 'C_t']]
#
# har_cj = get_RV_BV(data_filtered, alpha=0.05, times=False)
#
#
# # get returns
#
# def make_returns(prices):
#     """
#     Calculate returns from prices.
#     prices: array-like of prices
#     Returns: array-like of returns
#     """
#     n = len(prices)
#     returns = np.empty(n)
#     returns[0] = 0  # First return is 0
#
#     for i in range(1, n):
#         if prices[i - 1] == 0:
#             returns[i] = np.nan
#         else:
#             returns[i] = (prices[i] - prices[i - 1]) / prices[i - 1]
#
#     return returns
#
# # Load data
# # df = pd.read_csv("c:/Users/lenovo/Desktop/HAR/day_data.csv")
# df = pd.read_csv("c:/Users/lenovo/Desktop/HAR/沪深300/hs300.csv")
# # Rename columns and filter out missing values
# data_ret = (
#     df[['time', 'code', 'close']]  # Select relevant columns
#     .rename(columns={'time': 'DT', 'code': 'id', 'close': 'PRICE'})
#     .dropna()
#     .groupby('id')
#     .apply(lambda x: pd.DataFrame({
#         'DT': x['DT'],
#         'id': x['id'],
#         'Ret': make_returns(x['PRICE'].values)
#     }))
#     .reset_index(drop=True)
# )
#
# # Group summary
# group_summary = (
#     data_ret.groupby('id')
#     .size()
#     .reset_index(name='NumObservations')
# )
#
# # Filter for a specific group ('000001.XSHG')
# data_filtered = (
#     df[['time', 'code', 'close']]
#     .rename(columns={'time': 'DT', 'code': 'id', 'close': 'PRICE'})
#     .dropna()
#     .query("id == '000001.XSHG'")
# )
#
# # Calculate returns for the filtered data
# # Calculate returns for the filtered data
# returns = make_returns(data_filtered['PRICE'].values)
#
# # Create a DataFrame for returns with DT from the original filtered data
# returns_df = pd.DataFrame({
#     'DT': data_filtered['DT'],  # Use the original 'DT' column
#     'returns': returns
# })
#
# # Convert 'DT' to datetime format
# returns_df['DT'] = pd.to_datetime(returns_df['DT'])
#
#
#
# #
# # # 上证指数
# # df = pd.read_csv("data_idx.csv")

#
# #
# group_summary = df.groupby('code').size().reset_index(name='NumObservations')
#
#
# data_ret = df[['time', 'code', 'close']].copy()
# data_ret.columns = ['DT', 'id', 'PRICE']
# data_ret = data_ret.dropna()
#
# def calculate_returns(prices):
#     returns = prices.pct_change()
#     returns.iloc[0] = 0
#     returns[prices.shift(1) == 0] = np.nan
#     return returns
#
#
# data_ret['Ret'] = data_ret.groupby('id')['PRICE'].transform(calculate_returns)
#
#
# group_summary_ret = data_ret.groupby('id').size().reset_index(name='NumObservations')
#
#
# data_filtered = data_ret[data_ret['id'] == "000001.XSHG"].copy()
# data_filtered = data_filtered.drop('id', axis=1)
#
#
# data_filtered['DT'] = pd.to_datetime(data_filtered['DT']).dt.date
# RV = (data_filtered
#       .groupby('DT')['Ret']
#       .apply(lambda x: np.sum(x**2))
#       .reset_index())
#
# RV.columns = ['DT', 'RV']
#
#
# RV['DT'] = pd.to_datetime(RV['DT'])
#
# data_get_cj = pd.merge(RV, returns_df, on='DT', how='inner')
#
# data_get_cj = pd.merge(data_get_cj, har_cj, left_index=True, right_index=True)
#
# print(data_get_cj)
#
#
#





#   沪深300

#--------- 沪深300

os.chdir("c:/Users/lenovo/Desktop/HAR/沪深300")


df = pd.read_csv("hs300high.csv")


data_filtered = df[df['code'] == "000300.XSHG"].copy()


def get_RV_BV(data, alpha=0.05, times=True):

    idx = 100 if times else 1

    df = data.copy()


    df['datetime'] = pd.to_datetime(df['time'], format='%Y/%m/%d %H:%M')
    df['day'] = df['datetime'].dt.date

    results = []
    for day, group in df.groupby('day'):

        group = group.sort_values('datetime')


        group['Ret'] = (np.log(group['close']) - np.log(group['close'].shift(1))) * idx
        group['Ret_HL'] = (np.log(group['high']) - np.log(group['low'].shift(1))) * idx


        group = group.dropna(subset=['Ret', 'Ret_HL'])
        n = len(group)

        if n < 5:
            continue

        # 计算RV
        RV = np.sum(group['Ret'] ** 2)

        # 计算BV
        abs_ret = np.abs(group['Ret'])
        BV = (np.pi / 2) * np.sum(abs_ret.shift(1) * abs_ret.shift(-1).dropna())


        TQ_coef = n * (2 ** (2 / 3) * gamma(7 / 6) / gamma(0.5)) ** (-3) * (n / (n - 4))


        term1 = abs_ret.iloc[4:].values  # Ret[5:n()]
        term2 = abs_ret.iloc[2:-2].values  # Ret[3:(n-2)]
        term3 = abs_ret.iloc[:-4].values  # Ret[1:(n-4)]

        min_len = min(len(term1), len(term2), len(term3))
        if min_len > 0:
            TQ = TQ_coef * np.sum((term1[:min_len] ** (4 / 3)) *
                                  (term2[:min_len] ** (4 / 3)) *
                                  (term3[:min_len] ** (4 / 3)))
        else:
            continue

        # Z_test
        Z_test = ((RV - BV) / RV) / np.sqrt(((np.pi / 2) ** 2 + np.pi - 5) *
                                            (1 / n) * max(1, TQ / (BV ** 2)))

        # calculate JV
        q_alpha = norm.ppf(1 - alpha)
        JV = (RV - BV) * (Z_test > q_alpha)
        C_t = (Z_test <= q_alpha) * RV + (Z_test > q_alpha) * BV

        results.append({

            'BV': BV,
            'JV': JV,
            'C_t': C_t
        })


    result_df = pd.DataFrame(results)
    return result_df[['BV', 'JV', 'C_t']]

har_cj = get_RV_BV(data_filtered, alpha=0.05, times=False)


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
# df = pd.read_csv("c:/Users/lenovo/Desktop/HAR/day_data.csv")
df = pd.read_csv("c:/Users/lenovo/Desktop/HAR/沪深300/hs300.csv")
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
    .query("id == '000300.XSHG'")
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



#
# # 上证指数
# df = pd.read_csv("data_idx.csv")
df= pd.read_csv("hs300high.csv")

#
group_summary = df.groupby('code').size().reset_index(name='NumObservations')


data_ret = df[['time', 'code', 'close']].copy()
data_ret.columns = ['DT', 'id', 'PRICE']
data_ret = data_ret.dropna()

def calculate_returns(prices):
    returns = prices.pct_change()
    returns.iloc[0] = 0
    returns[prices.shift(1) == 0] = np.nan
    return returns


data_ret['Ret'] = data_ret.groupby('id')['PRICE'].transform(calculate_returns)


group_summary_ret = data_ret.groupby('id').size().reset_index(name='NumObservations')


data_filtered = data_ret[data_ret['id'] == "000300.XSHG"].copy()
data_filtered = data_filtered.drop('id', axis=1)


data_filtered['DT'] = pd.to_datetime(data_filtered['DT']).dt.date
RV = (data_filtered
      .groupby('DT')['Ret']
      .apply(lambda x: np.sum(x**2))
      .reset_index())

RV.columns = ['DT', 'RV']


RV['DT'] = pd.to_datetime(RV['DT'])

data_get_cj = pd.merge(RV, returns_df, on='DT', how='inner')

data_get_cj = pd.merge(data_get_cj, har_cj, left_index=True, right_index=True)



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




def rolling_mean(x, window):
    """Implements R's rollmean function"""
    return pd.Series(x).rolling(window=window, min_periods=1).mean()


def lag(x, n):
    """Implements R's lag function"""
    return pd.Series(x).shift(n)


# #
# def loss_function_vectorized(params, data, test_size=300):
#     try:
#         lanta1, lanta2, lanta3 = params
#
#         # 计算特征
#         r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, data['returns'][:i + 1].values, lanta1)
#                for i in range(len(data))]
#         cjt = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['JV'][:i + 1].values, lanta2)
#                for i in range(len(data))]
#         cvt = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['C_t'][:i + 1].values, lanta3)
#                for i in range(len(data))]
#
#         # 转换为Series
#         r2t = pd.Series(r2t)
#         cjt = pd.Series(cjt)
#         cvt = pd.Series(cvt)
#
#         # 创建特征矩阵
#         model_data = pd.DataFrame({
#             'RV': data['RV'],
#             'r2t_lag1': r2t.shift(1),
#             'r2t_lag5': r2t.rolling(window=5).mean(),
#             'r2t_lag22': r2t.rolling(window=22).mean(),
#             'cj_lag1': cjt.shift(1),
#             'cj_lag5': cjt.rolling(window=5).mean(),
#             'cj_lag22': cjt.rolling(window=22).mean(),
#             'cv_lag1': cvt.shift(1),
#             'cv_lag5': cvt.rolling(window=5).mean(),
#             'cv_lag22': cvt.rolling(window=22).mean()
#         })
#
#         # 删除缺失值
#         model_data = model_data.dropna().reset_index(drop=True)
#
#         # 划分训练集和测试集
#         train_data = model_data.iloc[:-test_size]
#         test_data = model_data.iloc[-test_size:]
#
#         # 准备数据
#         X_train = train_data.drop(columns='RV')
#         y_train = train_data['RV']
#         X_test = test_data.drop(columns='RV')
#         y_test = test_data['RV']
#
#         # 标准化特征
#         scaler_X = StandardScaler()
#         scaler_y = StandardScaler()
#
#         # 转换特征
#         X_train_scaled = scaler_X.fit_transform(X_train)
#         X_test_scaled = scaler_X.transform(X_test)
#
#         # 转换目标变量
#         y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
#         y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
#
#         # 训练模型
#         model = LinearRegression()
#         model.fit(X_train_scaled, y_train_scaled)
#
#         # 预测
#         y_pred_scaled = model.predict(X_test_scaled)
#
#         # 将预测值转换回原始尺度
#         y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
#
#         # 计算MSE
#         mse = np.mean((y_pred - y_test) ** 2)
#
#         return mse
#
#     except Exception as e:
#         print(f"Error in loss function: {e}")
#         return np.inf
#
#
# def optimize_parameters(data, test_size=300):
#     # 设置优化参数
#     de_result = differential_evolution(
#         func=loss_function_vectorized,
#         bounds=[(1e-6, 50)] * 3,
#         args=(data, test_size),
#         strategy='best1bin',
#         maxiter=200,
#         popsize=200,
#         tol=0.001,
#         disp=True,
#         workers=-1
#     )
#
#     # 输出结果
#     print("\n优化结果:")
#     print(f"最优参数值: lanta1={de_result.x[0]:.6f}, lanta2={de_result.x[1]:.6f}, lanta3={de_result.x[2]:.6f}")
#     print(f"最小MSE值: {de_result.fun:.6f}")
#     print(f"优化是否成功: {de_result.success}")
#     print(f"迭代次数: {de_result.nit}")
#
#     return de_result
# #
# #
# # # 主程序入口
# if __name__ == "__main__":
#     result = optimize_parameters(data_get_cj)


# lanta1 = 0.002636
# lanta2 = 24.224740
# lanta3 = 8.152140
# no lasso

#
# def loss_function_vectorized(params, data, test_size=300):
#     try:
#         lanta1, lanta2, lanta3 = params
#
#         # 计算特征
#         r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, data['returns'][:i + 1].values, lanta1)
#                for i in range(len(data))]
#         cjt = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['JV'][:i + 1].values, lanta2)
#                for i in range(len(data))]
#         cvt = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['C_t'][:i + 1].values, lanta3)
#                for i in range(len(data))]
#
#         # 转换为Series
#         r2t = pd.Series(r2t)
#         cjt = pd.Series(cjt)
#         cvt = pd.Series(cvt)
#
#         # 创建特征矩阵
#         model_data = pd.DataFrame({
#             'RV': data['RV'],
#             'r2t_lag1': r2t.shift(1),
#             'r2t_lag5': r2t.rolling(window=5).mean(),
#             'r2t_lag22': r2t.rolling(window=22).mean(),
#             'cj_lag1': cjt.shift(1),
#             'cj_lag5': cjt.rolling(window=5).mean(),
#             'cj_lag22': cjt.rolling(window=22).mean(),
#             'cv_lag1': cvt.shift(1),
#             'cv_lag5': cvt.rolling(window=5).mean(),
#             'cv_lag22': cvt.rolling(window=22).mean()
#         })
#
#         # 删除缺失值
#         model_data = model_data.dropna().reset_index(drop=True)
#
#         # 划分训练集和测试集
#         train_data = model_data.iloc[:-test_size]
#         test_data = model_data.iloc[-test_size:]
#
#         # 准备初始训练数据
#         X_train = train_data.drop(columns='RV')
#         y_train = train_data['RV']
#         X_test = test_data.drop(columns='RV')
#         y_test = test_data['RV']
#
#         # 初始化滚动窗口
#         rolling_X = X_train.copy()
#         rolling_y = y_train.copy()
#
#         predictions = []
#         squared_errors = []
#
#         # 滚动预测
#         for i in range(len(test_data)):
#             # 训练模型
#             model = LinearRegression()
#             model.fit(rolling_X, rolling_y)
#
#             # 预测下一个时间点
#             current_X = X_test.iloc[i:i + 1]
#             pred = model.predict(current_X)[0]
#             actual = y_test.iloc[i]
#
#             # 存储预测结果和误差
#             predictions.append(pred)
#             squared_errors.append((pred - actual) ** 2)
#
#             # 更新滚动窗口
#             if i < len(test_data) - 1:
#                 rolling_X = pd.concat([rolling_X.iloc[1:], current_X])
#                 rolling_y = pd.concat([rolling_y.iloc[1:], pd.Series([actual])])
#
#         # 计算MSE
#         mse = np.mean(squared_errors)
#
#         return mse
#
#     except Exception as e:
#         print(f"Error in loss function: {e}")
#         return np.inf
#
#
# def optimize_parameters(data, test_size=300):
#     # 设置优化参数
#     de_result = differential_evolution(
#         func=loss_function_vectorized,
#         bounds=[(1e-6, 40)] * 3,
#         args=(data, test_size),
#         strategy='best1bin',
#         maxiter=200,
#         popsize=50,
#         tol=0.0001,
#         disp=True,
#         workers=-1  # 使用并行计算
#     )
#
#     # 输出结果
#     print("\n优化结果:")
#     print(f"最优参数值: lanta1={de_result.x[0]:.6f}, lanta2={de_result.x[1]:.6f}, lanta3={de_result.x[2]:.6f}")
#     print(f"最小MSE值: {de_result.fun:.6f}")
#     print(f"优化是否成功: {de_result.success}")
#     print(f"迭代次数: {de_result.nit}")
#
#     return de_result
#
#
# # 主程序入口
# if __name__ == "__main__":
#     result = optimize_parameters(data_get_cj)
# #
#
#
# results_dict = {}
#
#
# def loss_function_vectorized(params, data, test_size=300):
#     try:
#         lanta1, lanta2, lanta3 = params
#
#         # 计算特征
#         r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, data['returns'][:i + 1].values, lanta1)
#                for i in range(len(data))]
#         cjt = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['JV'][:i + 1].values, lanta2)
#                for i in range(len(data))]
#         cvt = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['C_t'][:i + 1].values, lanta3)
#                for i in range(len(data))]
#
#         # 转换为Series
#         r2t = pd.Series(r2t)
#         cjt = pd.Series(cjt)
#         cvt = pd.Series(cvt)
#
#         # 创建特征矩阵
#         model_data = pd.DataFrame({
#             'RV': data['RV'],
#             'r2t_lag1': r2t.shift(1),
#             'r2t_lag5': r2t.rolling(window=5).mean(),
#             'r2t_lag22': r2t.rolling(window=22).mean(),
#             'cj_lag1': cjt.shift(1),
#             'cj_lag5': cjt.rolling(window=5).mean(),
#             'cj_lag22': cjt.rolling(window=22).mean(),
#             'cv_lag1': cvt.shift(1),
#             'cv_lag5': cvt.rolling(window=5).mean(),
#             'cv_lag22': cvt.rolling(window=22).mean()
#         })
#
#         # 删除缺失值
#         model_data = model_data.dropna().reset_index(drop=True)
#
#         # 划分训练集和测试集
#         train_data = model_data.iloc[:len(model_data) - test_size]
#         test_data = model_data.iloc[len(model_data) - test_size:]
#
#         # 准备训练数据
#         X_train = train_data.drop('RV', axis=1)
#         y_train = train_data['RV']
#         X_test = test_data.drop('RV', axis=1)
#         y_test = test_data['RV']
#
#         # 使用普通线性回归
#         linear_model = LinearRegression()
#         linear_model.fit(X_train, y_train)
#
#
#         # 在测试集上进行预测
#         y_pred = linear_model.predict(X_test)
#
#         # 计算MSE
#         mse = np.mean((y_pred - y_test) ** 2)
#
#         # 将参数和对应的特征信息存储到全局字典中
#         params_key = tuple(np.round(params, 6))
#         results_dict[params_key] = {
#             'significant_features': significant_features,
#             'insignificant_features': insignificant_features,
#             'coefficients': dict(zip(feature_names, coefficients))
#         }
#
#         return mse
#
#     except Exception as e:
#         print(f"Error in loss function: {e}")
#         return np.inf
#
#
# def hybrid_optimization(data):
#     # 使用差分进化进行优化
#     de_result = differential_evolution(
#         func=loss_function_vectorized,
#         bounds=[(1e-6, 100)] * 3,
#         args=(data,),
#         strategy='best1bin',
#         maxiter=300,
#         popsize=20,
#         tol=1e-6,
#         disp=True,
#         workers=-1
#     )
#
#     # 获取最优参数
#     optimal_params = de_result.x
#     print("\n最优参数值：")
#     print(f"lanta1 = {optimal_params[0]:.6f}")
#     print(f"lanta2 = {optimal_params[1]:.6f}")
#     print(f"lanta3 = {optimal_params[2]:.6f}")
#     print(f"最小MSE值：{de_result.fun}")
#
#     return de_result
#
#
# # 主程序入口
# if __name__ == "__main__":
#     best_result = hybrid_optimization(data_get_cj)
#



# #lasso
# results_dict = {}
# #
# def loss_function_vectorized(params, data, test_size=300):
#     try:
#         lanta1, lanta2, lanta3 = params
#
#         # 计算特征
#         r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, data['returns'][:i + 1].values, lanta1)
#                for i in range(len(data))]
#         cjt = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['JV'][:i + 1].values, lanta2)
#                for i in range(len(data))]
#         cvt = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['C_t'][:i + 1].values, lanta3)
#                for i in range(len(data))]
#
#         # 转换为Series
#         r2t = pd.Series(r2t)
#         cjt = pd.Series(cjt)
#         cvt = pd.Series(cvt)
#
#         # 创建特征矩阵
#         model_data = pd.DataFrame({
#             'RV': data['RV'],
#             'r2t_lag1': r2t.shift(1),
#             'r2t_lag5': r2t.rolling(window=5).mean(),
#             'r2t_lag22': r2t.rolling(window=22).mean(),
#             'cj_lag1': cjt.shift(1),
#             'cj_lag5': cjt.rolling(window=5).mean(),
#             'cj_lag22': cjt.rolling(window=22).mean(),
#             'cv_lag1': cvt.shift(1),
#             'cv_lag5': cvt.rolling(window=5).mean(),
#             'cv_lag22': cvt.rolling(window=22).mean()
#         })
#
#         # 删除缺失值
#         model_data = model_data.dropna().reset_index(drop=True)
#
#         # 划分训练集和测试集
#         train_data = model_data.iloc[:len(model_data) - test_size]
#         test_data = model_data.iloc[len(model_data) - test_size:]
#
#         # 准备训练数据
#         X_train = train_data.drop('RV', axis=1)
#         y_train = train_data['RV']
#         X_test = test_data.drop('RV', axis=1)
#         y_test = test_data['RV']
#
#         # 使用LassoCV进行交叉验证选择最优的alpha
#         lasso_cv = LassoCV(cv=10, max_iter=1000)
#         lasso_cv.fit(X_train, y_train)
#
#         # 使用最优alpha训练Lasso模型
#         alpha_optimal = lasso_cv.alpha_
#         lasso_model = Lasso(alpha=alpha_optimal)
#         lasso_model.fit(X_train, y_train)
#
#         # 获取非零系数的特征
#         coefficients = lasso_model.coef_
#         feature_names = X_train.columns
#         selected_features = feature_names[coefficients != 0]
#         penalized_features = feature_names[coefficients == 0]
#
#         # 在测试集上进行预测
#         y_pred = lasso_model.predict(X_test)
#
#         # 计算 MSE
#         mse = np.mean((y_pred - y_test) ** 2)
#
#         # 计算 L1 惩罚项
#         l1_penalty = alpha_optimal * np.sum(np.abs(coefficients))
#
#         # 将参数和对应的特征信息存储到全局字典中
#         params_key = tuple(np.round(params, 6))
#         results_dict[params_key] = {
#             'selected_features': selected_features.tolist(),
#             'penalized_features': penalized_features.tolist()
#         }
#
#         # 返回损失函数值
#         return mse + l1_penalty
#
#     except Exception as e:
#         # 出现异常时返回大的损失值
#         print(f"Error in loss function: {e}")
#         return np.inf
#
# # 优化函数
# def hybrid_optimization(data):
#         # 使用差分进化进行优化
#         de_result = differential_evolution(
#             func=loss_function_vectorized,
#             bounds=[(1e-6, 40)] * 3,
#             args=(data,),
#             strategy='best1bin',
#             maxiter=300,
#             popsize=100,
#             tol=1e-6,
#             disp=True,
#             workers=-1  # 使用并行计算
#         )
#
#         # 获取最优参数
#         optimal_params = de_result.x
#         print("\n最优参数值：")
#         print(f"lanta1 = {optimal_params[0]:.6f}")
#         print(f"lanta2 = {optimal_params[1]:.6f}")
#         print(f"lanta3 = {optimal_params[2]:.6f}")
#         print(f"最小损失函数值：{de_result.fun}")
#
#         # 根据最优参数的键获取对应的特征信息
#         params_key = tuple(np.round(optimal_params, 6))
#         optimal_features = results_dict.get(params_key, None)
#
#         if optimal_features is not None:
#             selected_features = optimal_features['selected_features']
#             penalized_features = optimal_features['penalized_features']
#         else:
#             # 如果找不到对应的特征信息，重新计算一次
#             loss_function_vectorized(optimal_params, data)
#             optimal_features = results_dict.get(params_key)
#             selected_features = optimal_features['selected_features']
#             penalized_features = optimal_features['penalized_features']
#
#         # 打印被惩罚和未被惩罚的特征
#         print("\n未被惩罚的特征（系数不为零）：")
#         print(", ".join(selected_features))
#
#         print("\n被惩罚的特征（系数为零）：")
#         print(", ".join(penalized_features))
#
#         return de_result
#
#     # 主程序入口
# if __name__ == "__main__":
#         best_result = hybrid_optimization(data_get_cj)

# #----------------------------------lasso600
# lanta1=2.91478626e-06
# lanta2=6.90962070e-01
# lanta3= 2.51720816e+01



#500  0.1154444   0.47761281 16.54580154
# 调用优化函数



#--------------------------600
#--------------no lasso600
#
# lanta1=2.44586110e-04
# lanta2=4.89865867e-01
# lanta3=2.97160548e+01

#-----------------------lasso
# lanta1 = 0.000701
# lanta2 = 56.876522
# lanta3 = 22.866751
# 被惩罚 r2tlag1, cvlag1,22 ,cjlag22
#

# #
# lanta1 = 0.000701
# lanta2 = 56.876522
# lanta3 = 22.866751





# # 800
# lanta1=0.000870
# lanta2=3.332478
# lanta3=21.946217


# 1000 no lasso
# lanta1=20.765965
# lanta2=38.833988
# lanta3=0.049535

# 1000 lasso
# lanta1=19.92956
# lanta2=10.26586
# lanta3=16.38951


# #  沪深300  no lasso

#
# lanta1 = 0.135102
# lanta2 = 0.032489
# lanta3 = 22.552070



r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, data_get_cj['returns'].iloc[:i+1].values, lanta1)
       for i in range(len(data_get_cj))]
cjt = [conv_fun1_vectorized(TSPL_kernel_vectorized, data_get_cj['JV'].iloc[:i+1].values, lanta2)
       for i in range(len(data_get_cj))]
cvt = [conv_fun1_vectorized(TSPL_kernel_vectorized, data_get_cj['C_t'].iloc[:i+1].values, lanta3)
       for i in range(len(data_get_cj))]
def check_nan(lst):
    for item in lst:
        if isinstance(item, (int, float)) and np.isnan(item):
            return True
        elif isinstance(item, list) and check_nan(item):
            return True
    return False

# 检查r2t中的NaN值
has_nan_r2t = check_nan(r2t)
print("r2t has NaN values:", has_nan_r2t)

# 检查cjt中的NaN值
has_nan_cjt = check_nan(cjt)
print("cjt has NaN values:", has_nan_cjt)

# 检查cvt中的NaN值
has_nan_cvt = check_nan(cvt)
print("cvt has NaN values:", has_nan_cvt)

r2t = pd.Series(r2t)
cjt = pd.Series(cjt)
cvt = pd.Series(cvt)
model_data = pd.DataFrame({
    'RV': RV['RV'],
    'r2t_lag1':r2t.shift(1),
    'r2t_lag5': r2t.rolling(window=5).mean(),
    'r2t_lag22': r2t.rolling(window=22).mean(),
    'cjt_lag1': cjt.shift(1),
    'cjt_lag5': cjt.rolling(window=5).mean(),
    'cjt_lag22': cjt.rolling(window=22).mean(),
    'cvt_lag1': cvt.shift(1),
    'cvt_lag5': cvt.rolling(window=5).mean(),
    'cvt_lag22': cvt.rolling(window=22).mean(),

})

#
# 删除包含缺失值的行
model_data = model_data.dropna()

# 定义因变量 y 和自变量 X
y = model_data['RV']
X = model_data.drop('RV', axis=1)

# 检查 X 和 y 是否为空
print(f"y is empty: {y.empty}")
print(f"X is empty: {X.empty}")

# 添加常数项
X = sm.add_constant(X)

# 确保没有零大小的数组
if y.empty or X.empty:
    print("Error: The data is empty or invalid.")
else:
    # 拟合线性回归模型
    model = sm.OLS(y, X).fit()

    # 获取并打印模型的详细统计摘要
    summary = model.summary()
    print(summary)


#
#
test_size = 300

# 划分训练集和测试集
train_data = model_data.iloc[:len(model_data) - test_size]
test_data = model_data.iloc[len(model_data) - test_size:]

# 分割特征和目标值
X_train = train_data.drop('RV', axis=1)
y_train = train_data['RV']
X_test = test_data.drop('RV', axis=1)
y_test = test_data['RV']

# Initialize prediction and actual value lists
predictions_lr1 = []
actuals_lr1 = []
predictions_lr5 = []
actuals_lr5 = []
predictions_lr22 = []
actuals_lr22 = []

# Initialize rolling window with training data
rolling_X = X_train.copy()
rolling_y = y_train.copy()

# Rolling window prediction
for i in range(len(X_test)):
    # Train model on current window
    model = LinearRegression()
    model.fit(rolling_X, rolling_y)

    # 1-step ahead prediction (单步预测)
    pred_1 = model.predict(X_test[i:i + 1])
    predictions_lr1.append(pred_1[0])
    actuals_lr1.append(y_test.iloc[i])

    # 5-step ahead prediction (5步预测)
    if i + 4 < len(X_test):
        pred_5 = model.predict(X_test.iloc[i:i + 5])
        predictions_lr5.append(pred_5[-1])
        actuals_lr5.append(y_test.iloc[i + 4])
    else:
        predictions_lr5.append(None)
        actuals_lr5.append(None)

    # 22-step ahead prediction (22步预测)
    if i + 21 < len(X_test):
        pred_22 = model.predict(X_test.iloc[i:i + 22])
        predictions_lr22.append(pred_22[-1])
        actuals_lr22.append(y_test.iloc[i + 21])
    else:
        predictions_lr22.append(None)
        actuals_lr22.append(None)

    # Update rolling window by removing oldest observation and adding new one
    rolling_X = np.vstack((rolling_X[1:], X_test.iloc[i:i + 1].values))
    rolling_y = np.concatenate((rolling_y[1:], [y_test.iloc[i]]))

# Create results DataFrame to store predictions and actuals
df_predictions_lr = pd.DataFrame({
    'Prediction_1': predictions_lr1,
    'Actual_1': actuals_lr1,
    'Prediction_5': predictions_lr5,
    'Actual_5': actuals_lr5,
    'Prediction_22': predictions_lr22,
    'Actual_22': actuals_lr22
})

# Print the first few rows to check the predictions
print(df_predictions_lr.head())

df_predictions_lr.to_csv('cj_pd7300.csv', index=False)
