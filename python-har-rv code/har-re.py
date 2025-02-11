import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from numba import jit
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler

from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LassoCV,Lasso
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import mstats
from scipy.optimize import minimize, differential_evolution
import statsmodels.formula.api as smf
from scipy.special import gamma
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import os
#  上上证指数
# os.chdir("c:/Users/lenovo/Desktop/HAR")
#
# # 读取数据
# df = pd.read_csv("data_idx.csv")
# # 按组进行分类统计
# group_summary = df.groupby('code').size().reset_index(name='NumObservations')
#
# # 筛选代码为 "000001.XSHG" 的数据
# data_filtered = df[df['code'] == "000001.XSHG"]
#
#
# def get_re(data, alpha):
#     # 将数据转换为DataFrame并确保是副本
#     result = data.copy()
#
#     # 打印时间列的几个样本，用于调试
#     print("Sample time values:", result['time'].head())
#
#     # 转换时间列 - 使用更健壮的方式处理日期
#     try:
#         # 如果时间格式是 "YYYY/M/D H" 这种格式
#         result['day'] = pd.to_datetime(result['time'], format='%Y/%m/%d %H')
#     except:
#         try:
#             # 如果上面的格式不工作，尝试其他常见格式
#             result['day'] = pd.to_datetime(result['time'])
#         except:
#             # 如果还是不行，尝试先分割时间字符串
#             result['day'] = pd.to_datetime(result['time'].str.split().str[0])
#
#     # 只保留日期部分
#     result['day'] = result['day'].dt.date
#
#     # 按天分组进行计算
#     def calculate_daily_metrics(group):
#         # 计算对数收益率
#         group['Ret'] = np.log(group['close']).diff()
#
#         # 删除缺失值
#         group = group.dropna()
#
#         if len(group) == 0:
#             return None
#
#         # 计算标准差
#         sigma = group['Ret'].std()
#
#         # 计算分位数阈值
#         r_minus = norm.ppf(alpha) * sigma
#         r_plus = norm.ppf(1 - alpha) * sigma
#
#         # 计算超额收益
#         REX_minus = np.sum(np.where(group['Ret'] <= r_minus, group['Ret'] ** 2, 0))
#         REX_plus = np.sum(np.where(group['Ret'] >= r_plus, group['Ret'] ** 2, 0))
#         REX_moderate = np.sum(np.where((group['Ret'] > r_minus) & (group['Ret'] < r_plus),
#                                        group['Ret'] ** 2, 0))
#
#         return pd.Series({
#             'REX_minus': REX_minus,
#             'REX_plus': REX_plus,
#             'REX_moderate': REX_moderate
#         })
#
#     # 按天分组计算指标
#     result = result.groupby('day').apply(calculate_daily_metrics).reset_index()
#
#     return result
#
#
#
# # 使用函数
# har_re = get_re(data_filtered, alpha=0.05)
#
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
# df = pd.read_csv("c:/Users/lenovo/Desktop/HAR/day_data.csv")
#
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
# # Read the data
# df = pd.read_csv("data_idx.csv")
#
# # Get group summary
# group_summary = df.groupby('code').size().reset_index(name='NumObservations')
#
# # Create data_ret DataFrame with renamed columns first
# data_ret = df[['time', 'code', 'close']].copy()
# data_ret.columns = ['DT', 'id', 'PRICE']
# data_ret = data_ret.dropna()
#
# # Calculate returns for each group
# def calculate_returns(prices):
#     returns = prices.pct_change()
#     returns.iloc[0] = 0
#     returns[prices.shift(1) == 0] = np.nan
#     return returns
#
# # Calculate returns by group
# data_ret['Ret'] = data_ret.groupby('id')['PRICE'].transform(calculate_returns)
#
# # Get group summary for data_ret
# group_summary_ret = data_ret.groupby('id').size().reset_index(name='NumObservations')
#
# # Filter for "000001.XSHG" and remove unnecessary columns
# data_filtered = data_ret[data_ret['id'] == "000001.XSHG"].copy()
# data_filtered = data_filtered.drop('id', axis=1)
#
# # Convert DT to datetime and calculate daily RV
# data_filtered['DT'] = pd.to_datetime(data_filtered['DT']).dt.date
# RV = (data_filtered
#       .groupby('DT')['Ret']
#       .apply(lambda x: np.sum(x**2))
#       .reset_index())
#
# # Ensure RV has the correct column names
# RV.columns = ['DT', 'RV']
#
# # Convert DT to datetime for consistency with har_cj
# RV['DT'] = pd.to_datetime(RV['DT'])
#
# # Merge RV and returns_df on 'DT'
# data_re = pd.merge(RV, returns_df, on='DT', how='inner')
# # Ensure 'har_cj' is properly prepared before merging
# har_pd_re = pd.merge(data_re, har_re, left_index=True, right_index=True)

# # # 计算滞后值和滚动均值
# # har_pd_re['rex_m_lag1'] = har_pd_re['REX_minus'].shift(1)
# # har_pd_re['rex_m_lag5'] = har_pd_re['REX_minus'].rolling(window=5, min_periods=1).mean()
# # har_pd_re['rex_m_lag22'] = har_pd_re['REX_minus'].rolling(window=22, min_periods=1).mean()
# #
# # har_pd_re['rex_p_lag1'] = har_pd_re['REX_plus'].shift(1)
# # har_pd_re['rex_p_lag5'] = har_pd_re['REX_plus'].rolling(window=5, min_periods=1).mean()
# # har_pd_re['rex_p_lag22'] = har_pd_re['REX_plus'].rolling(window=22, min_periods=1).mean()
# #
# # har_pd_re['rex_moderate_lag1'] = har_pd_re['REX_moderate'].shift(1)
# # har_pd_re['rex_moderate_lag5'] = har_pd_re['REX_moderate'].rolling(window=5, min_periods=1).mean()
# # har_pd_re['rex_moderate_lag22'] = har_pd_re['REX_moderate'].rolling(window=22, min_periods=1).mean()
# #
# # # 创建模型数据
# # model_data = har_pd_re[['RV', 'rex_m_lag1', 'rex_m_lag5', 'rex_m_lag22',
# #                         'rex_p_lag1', 'rex_p_lag5', 'rex_p_lag22',
# #                         'rex_moderate_lag1', 'rex_moderate_lag5', 'rex_moderate_lag22']]
# #
# # # 删除缺失值
# # model_data = model_data.dropna()
# #
# # # 拟合线性回归模型
# # model = ols('RV ~ rex_m_lag1 + rex_m_lag5 + rex_m_lag22 + rex_p_lag1 + rex_p_lag5 + rex_p_lag22 + rex_moderate_lag1 + rex_moderate_lag5 + rex_moderate_lag22', data=model_data).fit()
# #
# # # 输出模型摘要
# # print(model.summary())
# #
# # # 计算 AIC 和 BIC
# # model_aic = model.aic
# # model_bic = model.bic
# # print(f"AIC: {model_aic}")
# # print(f"BIC: {model_bic}")
# #
#
# #Define vectorized kernel functions

#  沪深300

os.chdir("c:/Users/lenovo/Desktop/HAR/沪深300")


df = pd.read_csv("hs300high.csv")

# 按组进行分类统计
group_summary = df.groupby('code').size().reset_index(name='NumObservations')

# 筛选代码为 "000001.XSHG" 的数据
data_filtered = df[df['code'] == "000300.XSHG"]


def get_re(data, alpha):
    # 将数据转换为DataFrame并确保是副本
    result = data.copy()

    # 打印时间列的几个样本，用于调试
    print("Sample time values:", result['time'].head())

    # 转换时间列 - 使用更健壮的方式处理日期
    try:
        # 如果时间格式是 "YYYY/M/D H" 这种格式
        result['day'] = pd.to_datetime(result['time'], format='%Y/%m/%d %H')
    except:
        try:
            # 如果上面的格式不工作，尝试其他常见格式
            result['day'] = pd.to_datetime(result['time'])
        except:
            # 如果还是不行，尝试先分割时间字符串
            result['day'] = pd.to_datetime(result['time'].str.split().str[0])

    # 只保留日期部分
    result['day'] = result['day'].dt.date

    # 按天分组进行计算
    def calculate_daily_metrics(group):
        # 计算对数收益率
        group['Ret'] = np.log(group['close']).diff()

        # 删除缺失值
        group = group.dropna()

        if len(group) == 0:
            return None

        # 计算标准差
        sigma = group['Ret'].std()

        # 计算分位数阈值
        r_minus = norm.ppf(alpha) * sigma
        r_plus = norm.ppf(1 - alpha) * sigma

        # 计算超额收益
        REX_minus = np.sum(np.where(group['Ret'] <= r_minus, group['Ret'] ** 2, 0))
        REX_plus = np.sum(np.where(group['Ret'] >= r_plus, group['Ret'] ** 2, 0))
        REX_moderate = np.sum(np.where((group['Ret'] > r_minus) & (group['Ret'] < r_plus),
                                       group['Ret'] ** 2, 0))

        return pd.Series({
            'REX_minus': REX_minus,
            'REX_plus': REX_plus,
            'REX_moderate': REX_moderate
        })

    # 按天分组计算指标
    result = result.groupby('day').apply(calculate_daily_metrics).reset_index()

    return result



# 使用函数
har_re = get_re(data_filtered, alpha=0.05)


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


df= pd.read_csv("hs300high.csv")

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
data_filtered = data_ret[data_ret['id'] == "000300.XSHG"].copy()
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
data_re = pd.merge(RV, returns_df, on='DT', how='inner')
# Ensure 'har_cj' is properly prepared before merging
har_pd_re = pd.merge(data_re, har_re, left_index=True, right_index=True)

print(har_pd_re)

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



#
def rolling_mean(x, window):
    """Implements R's rollmean function"""
    return pd.Series(x).rolling(window=window, min_periods=1).mean()


def lag(x, n):
    """Implements R's lag function"""
    return pd.Series(x).shift(n)
#
#

# 全局字典存储特征选择结果
results_dict = {}

# # 向量化损失函数
# def loss_function_vectorized(params, data, test_size=300):
#     global results_dict  # 使用全局变量存储结果
#     try:
#         lanta1, lanta2, lanta3, lanta4 = params
#
#         r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, data['returns'].iloc[:i + 1].values, lanta1)
#                for i in range(len(data))]
#         rex_p = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['REX_plus'].iloc[:i + 1].values, lanta2)
#                  for i in range(len(data))]
#         rex_m = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['REX_minus'].iloc[:i + 1].values, lanta3)
#                  for i in range(len(data))]
#         rex_d = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['REX_moderate'].iloc[:i + 1].values, lanta4)
#                  for i in range(len(data))]
#
#         # 转换为 Series
#         r2t = pd.Series(r2t)
#         rex_m = pd.Series(rex_m)
#         rex_p = pd.Series(rex_p)
#         rex_d = pd.Series(rex_d)
#
#         # 创建特征矩阵
#         model_data = pd.DataFrame({
#             'RV': data['RV'],
#             'r2t_lag1': pd.Series(r2t).shift(1),
#             'r2t_lag5': pd.Series(r2t).rolling(window=5).mean(),
#             'r2t_lag22': pd.Series(r2t).rolling(window=22).mean(),
#             'rex_minus_lag1': pd.Series(rex_m).shift(1),
#             'rex_minus_lag5': pd.Series(rex_m).rolling(window=5).mean(),
#             'rex_minus_lag22': pd.Series(rex_m).rolling(window=22).mean(),
#             'rex_plus_lag1': pd.Series(rex_p).shift(1),
#             'rex_plus_lag5': pd.Series(rex_p).rolling(window=5).mean(),
#             'rex_plus_lag22': pd.Series(rex_p).rolling(window=22).mean(),
#             'rex_moderate_lag1': pd.Series(rex_d).shift(1),
#             'rex_moderate_lag5': pd.Series(rex_d).rolling(window=5).mean(),
#             'rex_moderate_lag22': pd.Series(rex_d).rolling(window=22).mean()
#         }).dropna().reset_index(drop=True)
#
#         # 划分训练集和测试集
#         train_data = model_data.iloc[:-test_size]
#         test_data = model_data.iloc[-test_size:]
#
#         X_train = train_data.drop('RV', axis=1)
#         y_train = train_data['RV']
#         X_test = test_data.drop('RV', axis=1)
#         y_test = test_data['RV']
#
#         # 标准化特征
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#
#         # 使用简单的线性回归模型进行预测
#         lr_model = LinearRegression()
#         lr_model.fit(X_train_scaled, y_train)
#
#         # 在测试集上进行预测
#         y_pred = lr_model.predict(X_test_scaled)
#
#         # 计算 MSE
#         mse = np.mean((y_pred - y_test) ** 2)
#
#         # 将参数和对应的 MSE 存储到全局字典中
#         params_key = tuple(np.round(params, 6))
#         results_dict[params_key] = {
#             'mse': mse
#         }
#
#         # 返回损失函数值
#         return mse
#
#     except Exception as e:
#         # 出现异常时返回大的损失值
#         print(f"Error in loss function: {e}")
#         return np.inf
#
# # 优化函数
# def hybrid_optimization(data):
#     # 使用差分进化进行优化
#     de_result = differential_evolution(
#         func=loss_function_vectorized,
#         bounds=[(1e-6, 50)] * 4,  # 缩小参数范围
#         args=(data,),
#         strategy='rand1bin',  # 尝试不同策略
#         maxiter=100,  # 增大最大迭代次数
#         popsize=80,  # 增加种群规模
#         tol=0.001,
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
#     print(f"lanta4 = {optimal_params[3]:.6f}")
#     print(f"最小损失函数值：{de_result.fun}")
#
#     return de_result
#
# # 主程序入口
# if __name__ == "__main__":
#     best_result = hybrid_optimization(har_pd_re)


# # 全局字典存储特征选择结果
# results_dict = {}
#
#
# # 向量化损失函数
# def loss_function_vectorized(params, data, test_size=300):
#     global results_dict  # 使用全局变量存储结果
#     try:
#         lanta1, lanta2, lanta3, lanta4 = params
#
#         r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, data['returns'].iloc[:i + 1].values, lanta1)
#                for i in range(len(data))]
#         rex_p = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['REX_plus'].iloc[:i + 1].values, lanta2)
#                  for i in range(len(data))]
#         rex_m = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['REX_minus'].iloc[:i + 1].values, lanta3)
#                  for i in range(len(data))]
#         rex_d = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['REX_moderate'].iloc[:i + 1].values, lanta4)
#                  for i in range(len(data))]
#
#         # 转换为 Series
#         r2t = pd.Series(r2t)
#         rex_m = pd.Series(rex_m)
#         rex_p = pd.Series(rex_p)
#         rex_d = pd.Series(rex_d)
#
#         # 创建特征矩阵
#         model_data = pd.DataFrame({
#             'RV': data['RV'],
#             'r2t_lag1': pd.Series(r2t).shift(1),
#             'r2t_lag5': pd.Series(r2t).rolling(window=5).mean(),
#             'r2t_lag22': pd.Series(r2t).rolling(window=22).mean(),
#             'rex_minus_lag1': pd.Series(rex_m).shift(1),
#             'rex_minus_lag5': pd.Series(rex_m).rolling(window=5).mean(),
#             'rex_minus_lag22': pd.Series(rex_m).rolling(window=22).mean(),
#             'rex_plus_lag1': pd.Series(rex_p).shift(1),
#             'rex_plus_lag5': pd.Series(rex_p).rolling(window=5).mean(),
#             'rex_plus_lag22': pd.Series(rex_p).rolling(window=22).mean(),
#             'rex_moderate_lag1': pd.Series(rex_d).shift(1),
#             'rex_moderate_lag5': pd.Series(rex_d).rolling(window=5).mean(),
#             'rex_moderate_lag22': pd.Series(rex_d).rolling(window=22).mean()
#         }).dropna().reset_index(drop=True)
#
#         # 划分训练集和测试集
#         train_data = model_data.iloc[:-test_size]
#         test_data = model_data.iloc[-test_size:]
#
#         X_train = train_data.drop('RV', axis=1)
#         y_train = train_data['RV']
#         X_test = test_data.drop('RV', axis=1)
#         y_test = test_data['RV']
#
#         # 标准化特征
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#
#         # 使用 LassoCV 进行特征选择和模型拟合
#         lasso_cv = LassoCV(cv=10, n_alphas=50, max_iter=1000, tol=1e-4)
#         lasso_cv.fit(X_train_scaled, y_train)
#
#         # 使用最优 alpha 训练 Lasso 模型
#         alpha_optimal = lasso_cv.alpha_
#         lasso_model = Lasso(alpha=alpha_optimal)
#         lasso_model.fit(X_train_scaled, y_train)
#
#         # 获取非零系数的特征
#         coefficients = lasso_model.coef_
#         feature_names = X_train.columns
#         selected_features = feature_names[coefficients != 0]
#         penalized_features = feature_names[coefficients == 0]
#
#         # 在测试集上进行预测
#         y_pred = lasso_model.predict(X_test_scaled)
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
#     # 使用差分进化进行优化
#     de_result = differential_evolution(
#         func=loss_function_vectorized,
#         bounds=[(1e-6, 50)] * 4,  # 缩小参数范围
#         args=(data,),
#         strategy='rand1bin',  # 尝试不同策略
#         maxiter=100,  # 增大最大迭代次数
#         popsize=40,  # 增加种群规模
#         tol=0.001,
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
#     print(f"lanta4 = {optimal_params[3]:.6f}")
#     print(f"最小损失函数值：{de_result.fun}")
#
#     # 根据最优参数的键获取对应的特征信息
#     params_key = tuple(np.round(optimal_params, 6))
#     optimal_features = results_dict.get(params_key, None)
#
#     if optimal_features is not None:
#         selected_features = optimal_features['selected_features']
#         penalized_features = optimal_features['penalized_features']
#     else:
#         # 如果找不到对应的特征信息，重新计算一次
#         loss_function_vectorized(optimal_params, data)
#         optimal_features = results_dict.get(params_key)
#         selected_features = optimal_features['selected_features']
#         penalized_features = optimal_features['penalized_features']
#
#     # 打印被惩罚和未被惩罚的特征
#     print("\n未被惩罚的特征（系数不为零）：")
#     print(", ".join(selected_features))
#
#     print("\n被惩罚的特征（系数为零）：")
#     print(", ".join(penalized_features))
#
#     return de_result
#
# # 主程序入口
# if __name__ == "__main__":
#     best_result = hybrid_optimization(har_pd_re)






# r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, har_pd_re['returns'].iloc[:i+1].values, lanta1)
#        for i in range(len(har_pd_re))]
# rex_p = [conv_fun1_vectorized(TSPL_kernel_vectorized, har_pd_re['REX_plus'].iloc[:i+1].values, lanta2)
#        for i in range(len(har_pd_re))]
# rex_m = [conv_fun1_vectorized(TSPL_kernel_vectorized, har_pd_re['REX_minus'].iloc[:i+1].values, lanta3)
#        for i in range(len(har_pd_re))]
# rex_d = [conv_fun1_vectorized(TSPL_kernel_vectorized, har_pd_re['REX_moderate'].iloc[:i+1].values, lanta4)
#        for i in range(len(har_pd_re))]
#
# def check_nan(lst):
#     for item in lst:
#         if isinstance(item, (int, float)) and np.isnan(item):
#             return True
#         elif isinstance(item, list) and check_nan(item):
#             return True
#     return False
#
#
# r2t= pd.Series(r2t)
# rex_m = pd.Series(rex_m)
# rex_p = pd.Series(rex_p)
# rex_d = pd.Series(rex_d)
# #
# # # 创建 DataFrame
#
# model_data = pd.DataFrame({
#     'RV': har_pd_re['RV'],
#     'r2t_lag1': r2t.shift(1),
#     'r2t_lag5': r2t.rolling(window=5).mean(),
#     'rex_minus_lag1': rex_m.shift(1),
#     'rex_minus_lag5': rex_m.rolling(window=5).mean(),
#     'rex_plus_lag1': rex_p.shift(1),
#     'rex_plus_lag5': rex_p.rolling(window=5).mean(),
#     'rex_moderate_lag1': rex_d.shift(1),
#     'rex_moderate_lag5': rex_d.rolling(window=5).mean()
# })
#
#
#
# # 去除可能的缺失值（如果需要）
# model_data = model_data.dropna().reset_index(drop=True)
#
# print(model_data)
#
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

# test_size = 600
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
# df_predictions_lr.to_csv('re_pdlasso600new.csv', index=False)
#

#
#
#
# no lasso
# def loss_function_vectorized(params, data, test_size=300):
#     """
#     只预测一步的MSE损失函数
#     """
#     try:
#         lanta1, lanta2, lanta3, lanta4 = params
#
#         # 计算特征
#         r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, data['returns'].iloc[:i + 1].values, lanta1)
#                for i in range(len(data))]
#         rex_p = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['REX_plus'].iloc[:i + 1].values, lanta2)
#                  for i in range(len(data))]
#         rex_m = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['REX_minus'].iloc[:i + 1].values, lanta3)
#                  for i in range(len(data))]
#         rex_d = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['REX_moderate'].iloc[:i + 1].values, lanta4)
#                  for i in range(len(data))]
#
#         # 转换为Series
#         r2t = pd.Series(r2t)
#         rex_m = pd.Series(rex_m)
#         rex_p = pd.Series(rex_p)
#         rex_d = pd.Series(rex_d)
#
#         # 创建特征矩阵
#         model_data = pd.DataFrame({
#             'RV': data['RV'],
#             'r2t_lag1': r2t.shift(1),
#             'r2t_lag5': r2t.rolling(window=5).mean(),
#             'r2t_lag22': r2t.rolling(window=22).mean(),
#             'rex_minus_lag1': rex_m.shift(1),
#             'rex_minus_lag5': rex_m.rolling(window=5).mean(),
#             'rex_minus_lag22': rex_m.rolling(window=22).mean(),
#             'rex_plus_lag1': rex_p.shift(1),
#             'rex_plus_lag5': rex_p.rolling(window=5).mean(),
#             'rex_plus_lag22': rex_p.rolling(window=22).mean(),
#             'rex_moderate_lag1': rex_d.shift(1),
#             'rex_moderate_lag5': rex_d.rolling(window=5).mean(),
#             'rex_moderate_lag22': rex_d.rolling(window=22).mean()
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
#         # 初始化预测误差列表
#         mse_steps = []
#
#         # 使用滚动窗口进行预测
#         rolling_X = X_train.copy()
#         rolling_y = y_train.copy()
#
#         for i in range(len(X_test)):
#             # 训练模型
#             model = LinearRegression()
#             model.fit(rolling_X, rolling_y)
#
#             # 一步预测
#             pred = model.predict(X_test[i:i + 1])
#             mse_steps.append((pred[0] - y_test.iloc[i]) ** 2)
#
#             # 更新滚动窗口
#             rolling_X = np.vstack((rolling_X[1:], X_test.iloc[i:i + 1].values))
#             rolling_y = np.concatenate((rolling_y[1:], [y_test.iloc[i]]))
#
#         # 计算MSE
#         mse = np.mean(mse_steps)
#
#         return mse
#
#     except Exception as e:
#         print(f"Error in loss function: {str(e)}")
#         return np.inf
#
#
# def optimize_parameters(data, test_size=300):
#     # 设置优化参数
#     de_result = differential_evolution(
#         func=loss_function_vectorized,
#         bounds=[(1e-6, 40)] * 4,
#         args=(data, test_size),
#         strategy='best1bin',
#         maxiter=300,
#         popsize=40,
#         tol=0.0001,
#         disp=True,
#         workers=-1  # 使用并行计算
#     )
#
#     # 输出结果
#     print("\n优化结果:")
#     print(f"最优参数值: lanta1={de_result.x[0]:.6f}, lanta2={de_result.x[1]:.6f}, lanta3={de_result.x[2]:.6f},lanta4={de_result.x[3]:.6f}")
#     print(f"最小MSE值: {de_result.fun:.6f}")
#     print(f"优化是否成功: {de_result.success}")
#     print(f"迭代次数: {de_result.nit}")
#
#     return de_result
#
# # 主程序入口
# if __name__ == "__main__":
#     result = optimize_parameters(har_pd_re)
# #
# # # 使用最优参数重新计算特征和预测结果
# # lanta1, lanta2, lanta3, lanta4 = best_result.x
# #


r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, har_pd_re['returns'].iloc[:i+1].values, lanta1)
       for i in range(len(har_pd_re))]
rex_p = [conv_fun1_vectorized(TSPL_kernel_vectorized, har_pd_re['REX_plus'].iloc[:i+1].values, lanta2)
       for i in range(len(har_pd_re))]
rex_m = [conv_fun1_vectorized(TSPL_kernel_vectorized, har_pd_re['REX_minus'].iloc[:i+1].values, lanta3)
       for i in range(len(har_pd_re))]
rex_d = [conv_fun1_vectorized(TSPL_kernel_vectorized, har_pd_re['REX_moderate'].iloc[:i+1].values, lanta4)
       for i in range(len(har_pd_re))]

def check_nan(lst):
    for item in lst:
        if isinstance(item, (int, float)) and np.isnan(item):
            return True
        elif isinstance(item, list) and check_nan(item):
            return True
    return False



#
r2t= pd.Series(r2t)
rex_m = pd.Series(rex_m)
rex_p = pd.Series(rex_p)
rex_d = pd.Series(rex_d)
#
# # 创建 DataFrame

model_data = pd.DataFrame({
    'RV': har_pd_re['RV'],
    'r2t_lag1': r2t.shift(1),
    'r2t_lag5': r2t.rolling(window=5).mean(),
    'r2t_lag22': r2t.rolling(window=22).mean(),
    'rex_minus_lag1': rex_m.shift(1),
    'rex_minus_lag5': rex_m.rolling(window=5).mean(),
    'rex_minus_lag22': rex_m.rolling(window=22).mean(),
    'rex_plus_lag1': rex_p.shift(1),
    'rex_plus_lag5': rex_p.rolling(window=5).mean(),
    'rex_plus_lag22': rex_p.rolling(window=22).mean(),
    'rex_moderate_lag1': rex_d.shift(1),
    'rex_moderate_lag5': rex_d.rolling(window=5).mean(),
    'rex_moderate_lag22': rex_d.rolling(window=22).mean()
})



# 去除可能的缺失值（如果需要）
model_data = model_data.dropna().reset_index(drop=True)

print(model_data)


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

df_predictions_lr.to_csv('re_pd7300.csv', index=False)



