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

import statsmodels.formula.api as smf
from scipy.special import gamma
from scipy.stats import norm
from tqdm import tqdm

from scipy.optimize import shgo
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Lasso

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
import os


from pathlib import Path

#------------------------------ HAR-RS 模型
# 加载数据
df = pd.read_csv("c:/Users/lenovo/Desktop/HAR/day_data.csv")



def calculate_returns(prices):
    """Calculate returns from price series."""
    returns = np.zeros(len(prices))
    for i in range(1, len(prices)):
        if prices[i - 1] == 0:
            returns[i] = np.nan
        else:
            returns[i] = (prices[i] - prices[i - 1]) / prices[i - 1]
    returns[0] = 0
    return returns


def calculate_RS(data):
    """Calculate RS+ and RS- from returns."""
    positive_returns = np.where(data['Ret'] > 0, data['Ret'], 0)
    negative_returns = np.where(data['Ret'] < 0, data['Ret'], 0)

    RS_plus = np.sum(np.square(positive_returns))
    RS_minus = np.sum(np.square(negative_returns))

    return pd.Series({
        'RS_plus': RS_plus,
        'RS_minus': RS_minus
    })


def process_har_rs_model(day_data_path, data_idx_path):
    """Process data for HAR-RS model."""
    # Read the data
    df = pd.read_csv(day_data_path)

    # Process first dataset
    data_ret = (
        df[['time', 'code', 'close']]
        .rename(columns={'time': 'DT', 'code': 'id', 'close': 'PRICE'})
        .dropna()
    )

    # Calculate returns for each group
    grouped = data_ret.groupby('id')
    returns_list = []

    for name, group in grouped:
        group_returns = pd.DataFrame({
            'DT': group['DT'],
            'id': group['id'],
            'Ret': calculate_returns(group['PRICE'].values)
        })
        returns_list.append(group_returns)

    data_ret = pd.concat(returns_list, ignore_index=True)

    # Group summary
    group_summary = data_ret.groupby('id').size().reset_index(name='NumObservations')

    # Filter for specific ID
    data_filtered = (
        df[['time', 'code', 'close']]
        .rename(columns={'time': 'DT', 'code': 'id', 'close': 'price'})
        .dropna()
        .query('id == "000001.XSHG"')
    )

    returns = calculate_returns(data_filtered['price'].values)

    # Process second dataset
    df_idx = pd.read_csv(data_idx_path)
    data_ret_idx = (
        df_idx[['time', 'code', 'close']]
        .rename(columns={'time': 'DT', 'code': 'id', 'close': 'PRICE'})
        .dropna()
    )

    # Calculate returns for index data
    grouped_idx = data_ret_idx.groupby('id')
    returns_list_idx = []

    for name, group in grouped_idx:
        group_returns = pd.DataFrame({
            'DT': group['DT'],
            'id': group['id'],
            'Ret': calculate_returns(group['PRICE'].values)
        })
        returns_list_idx.append(group_returns)

    data_ret_idx = pd.concat(returns_list_idx, ignore_index=True)

    # Filter for specific index
    data_cj = data_ret_idx.query('id == "000001.XSHG"').copy()

    # Calculate RS statistics by date
    result = (
        data_cj.groupby(pd.to_datetime(data_cj['DT']).dt.date)
        .apply(calculate_RS)
        .reset_index()
    )

    # Combine results
    final_data = pd.concat([
        result,
        pd.DataFrame({'returns': returns})
    ], axis=1)

    return final_data, group_summary


# Usage example:
if __name__ == "__main__":
    day_data_path = Path("c:/Users/lenovo/Desktop/HAR/day_data.csv")
    data_idx_path = Path("c:/Users/lenovo/Desktop/HAR/data_idx.csv")

    final_data, group_summary = process_har_rs_model(day_data_path, data_idx_path)
    print("Group Summary:")
    print(group_summary)
    print("\nFinal Data Sample:")
    print(final_data.head())



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

# Ensure both 'DT' columns are datetime
RV['DT'] = pd.to_datetime(RV['DT'])
final_data['DT'] = pd.to_datetime(final_data['DT'])

# Merge the dataframes
data_rs = pd.merge(RV, final_data, on='DT', how='inner')

# Display the merged dataframe
print(data_rs.head())



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
#lasso---------------------------
#
#
# def r2_loss_function_cv(params, data, test_size=600):
#     """
#     使用LassoCV优化的R²损失函数，通过交叉验证自动选择最优alpha
#     """
#     global best_nonnegative_params, best_r2_score
#
#     try:
#         lanta1, lanta2, lanta3 = params
#
#         # 计算特征
#         r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, data['returns'][:i + 1].values, lanta1)
#                for i in range(len(data))]
#         rs_p = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['RS_plus'][:i + 1].values, lanta2)
#                 for i in range(len(data))]
#         rs_m = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['RS_minus'][:i + 1].values, lanta3)
#                 for i in range(len(data))]
#
#         # 创建特征矩阵
#         model_data = pd.DataFrame({
#             'RV': data['Ret'],
#             'r2t_lag1': pd.Series(r2t).shift(1),
#             'r2t_lag5': pd.Series(r2t).rolling(window=5).mean(),
#             'r2t_lag22': pd.Series(r2t).rolling(window=22).mean(),
#             'rs_p_lag1': pd.Series(rs_p).shift(1),
#             'rs_p_lag5': pd.Series(rs_p).rolling(window=5).mean(),
#             'rs_p_lag22': pd.Series(rs_p).rolling(window=22).mean(),
#             'rs_m_lag1': pd.Series(rs_m).shift(1),
#             'rs_m_lag5': pd.Series(rs_m).rolling(window=5).mean(),
#             'rs_m_lag22': pd.Series(rs_m).rolling(window=22).mean(),
#             'rs_ratio': pd.Series(rs_p) / (pd.Series(rs_m) + 1e-6),
#             'rs_diff': pd.Series(rs_p) - pd.Series(rs_m)
#         }).dropna()
#
#         # 划分数据
#         train_data = model_data.iloc[:-test_size]
#         test_data = model_data.iloc[-test_size:]
#         X_train, y_train = train_data.drop(columns='RV'), train_data['RV']
#         X_test, y_test = test_data.drop(columns='RV'), test_data['RV']
#
#         # 标准化特征
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#
#         # 使用LassoCV进行交叉验证
#         lasso_cv = LassoCV(
#             cv=5,  # 5折交叉验证
#             n_alphas=100,  # 测试100个alpha值
#             max_iter=2000,
#             tol=1e-4,
#             random_state=42,
#             selection='random'  # 使用随机选择策略加速
#         )
#
#         # 在训练集上拟合LassoCV
#         lasso_cv.fit(X_train_scaled, y_train)
#
#         # 获取最优alpha值
#         optimal_alpha = lasso_cv.alpha_
#
#         # 使用最优alpha的Lasso模型在测试集上预测
#         y_pred = lasso_cv.predict(X_test_scaled)
#
#         # 计算R²分数
#         from sklearn.metrics import r2_score
#         r2 = r2_score(y_test, y_pred)
#
#         # 计算调整后的R²
#         n = len(y_test)
#         p = X_test.shape[1]
#         adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
#
#         # 计算Lasso惩罚项
#         lasso_penalty = optimal_alpha * np.sum(np.abs(lasso_cv.coef_))
#
#         # 如果存在负值预测，返回大的损失值
#         if np.any(y_pred < 0):
#             return np.inf
#
#         # 计算最终得分
#         final_score = -(adjusted_r2 - lasso_penalty)
#
#         # 更新最优解
#         if -final_score > best_r2_score:
#             best_r2_score = -final_score
#             best_nonnegative_params = params
#
#             # 打印详细信息
#             print("\n新的最优解找到:")
#             print(f"参数 (lanta1, lanta2, lanta3): {params}")
#             print(f"最优 alpha: {optimal_alpha:.6f}")
#             print(f"调整R²: {adjusted_r2:.4f}")
#             print(f"Lasso惩罚: {lasso_penalty:.4f}")
#             print("\n特征重要性:")
#             importance = pd.DataFrame({
#                 'feature': X_train.columns,
#                 'coefficient': lasso_cv.coef_
#             }).sort_values('coefficient', key=abs, ascending=False)
#
#             # 只打印非零系数的特征
#             non_zero_features = importance[abs(importance['coefficient']) > 1e-4]
#             print(non_zero_features.to_string(index=False))
#             print("-" * 50)
#
#         return final_score
#
#     except Exception as e:
#         print(f"Error in loss function: {str(e)}")
#         return np.inf
#
#
# def r2_optimization_cv(data):
#     """
#     使用交叉验证的R²优化过程
#     """
#     global best_nonnegative_params, best_r2_score
#     best_nonnegative_params = None
#     best_r2_score = -np.inf
#
#     # 使用差分进化进行优化
#     de_result = differential_evolution(
#         func=r2_loss_function_cv,
#         bounds=[(1e-6, 40)] * 3,
#         args=(data,),
#         strategy='best1bin',
#         maxiter=150,
#         popsize=25,
#         tol=0.001,
#         mutation=(0.5, 1.0),
#         recombination=0.7,
#         disp=True,
#         workers=-1
#     )
#
#     print("\n最终优化结果:")
#     print(f"最优参数: {best_nonnegative_params}")
#     print(f"最佳调整R²: {best_r2_score:.4f}")
#
#     return de_result
#
# #使用示例
# best_result = r2_optimization_cv(data_rs)
# lanta1,lanta2,lanta3= best_result.x

#

#
r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, data_rs['returns'][:i + 1].values, lanta1)
       for i in range(len(data_rs))]
rs_p = [conv_fun1_vectorized(TSPL_kernel_vectorized, data_rs['RS_plus'][:i + 1].values, lanta2)
        for i in range(len(data_rs))]
rs_m = [conv_fun1_vectorized(TSPL_kernel_vectorized, data_rs['RS_minus'][:i + 1].values, lanta3)
        for i in range(len(data_rs))]
r2t = pd.Series(r2t)
rs_p = pd.Series(rs_p)
rs_m = pd.Series(rs_m)

model_data = pd.DataFrame({
            'RV': data_rs['Ret'],
            'r2t_lag1': r2t.shift(1),
            'r2t_lag5': r2t.rolling(window=5).mean(),
            'r2t_lag22': r2t.rolling(window=22).mean(),
            'rs_p_lag1': rs_p.shift(1),
            'rs_p_lag5': rs_p.rolling(window=5).mean(),
            'rs_p_lag22': rs_p.rolling(window=22).mean(),
            'rs_m_lag1': rs_m.shift(1),
            'rs_m_lag5': rs_m.rolling(window=5).mean(),
            'rs_m_lag22': rs_m.rolling(window=22).mean()
})


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




test_size = 600

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
#
# df_predictions_lr.to_csv('rs_pd600.csv', index=False)
