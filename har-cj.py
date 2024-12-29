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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import os


os.chdir("c:/Users/lenovo/Desktop/HAR")


df = pd.read_csv("data_idx.csv")


data_filtered = df[df['code'] == "000001.XSHG"].copy()



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




# Read the data
df = pd.read_csv("data_idx.csv")

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
data_get_cj = pd.merge(RV, returns_df, on='DT', how='inner')

# Ensure 'har_cj' is properly prepared before merging
data_get_cj = pd.merge(data_get_cj, har_cj, left_index=True, right_index=True)
print(data_get_cj)


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


def loss_function_vectorized(params, data):
    """Vectorized loss function for optimization"""
    lanta1, lanta2, lanta3 = params

    # Calculate measures using the previously defined conv functions
    r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, data['returns'].iloc[:i + 1].values, lanta1)
           for i in range(len(data))]

    cjt = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['JV'].iloc[:i + 1].values, lanta2)
           for i in range(len(data))]

    cvt = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['C_t'].iloc[:i + 1].values, lanta3)
           for i in range(len(data))]

    # Create DataFrame with all necessary variables
    model_data = pd.DataFrame({
        'RV': data['RV'],
        'r2t_lag1': lag(r2t, 1),
        'r2t_lag5': rolling_mean(r2t, 5),
        'r2t_lag22': rolling_mean(r2t, 22),
        'cjt_lag1': lag(cjt, 1),
        'cjt_lag5': rolling_mean(cjt, 5),
        'cjt_lag22': rolling_mean(cjt, 22),
        'cvt_lag1': lag(cvt, 1),
        'cvt_lag5': rolling_mean(cvt, 5),
        'cvt_lag22': rolling_mean(cvt, 22)
    })

    # Remove NaN values
    model_data = model_data.dropna()

    if len(model_data) < 2:
        return np.inf

    # Prepare X and y for linear regression
    y = model_data['RV']
    X = model_data.drop('RV', axis=1)

    # Add constant term for intercept
    X = sm.add_constant(X)

    try:
        # Fit linear regression model
        model = sm.OLS(y, X).fit()
        adj_r_squared = model.rsquared_adj

        # Return negative adj R-squared for minimization
        return -adj_r_squared
    except:
        return np.inf


# Set up optimization
initial_params = np.array([1., 1., 1.])  # Initial lambda values

# Define bounds
bounds = [(1e-6, 30), (1e-6, 30), (1e-6, 30)]

# Run optimization
opt_result = minimize(
    loss_function_vectorized,
    initial_params,
    args=(data_get_cj,),
    method='L-BFGS-B',
    bounds=bounds,
    options={
        'maxiter': 1000,
        'disp': True
    }
)


# lasso--------------------------------------
#
# # Create model matrix
# def create_model_matrix(data, r2t, cjt, cvt):
#     model_data = pd.DataFrame({
#         'RV': data['RV'],
#         'r2t_lag1': np.roll(r2t, 1),
#         'r2t_lag5': uniform_filter1d(r2t, size=5, mode='nearest'),
#         'r2t_lag22': uniform_filter1d(r2t, size=22, mode='nearest'),
#         'cjt_lag1': np.roll(cjt, 1),
#         'cjt_lag5': uniform_filter1d(cjt, size=5, mode='nearest'),
#         'cjt_lag22': uniform_filter1d(cjt, size=22, mode='nearest'),
#         'cvt_lag1': np.roll(cvt, 1),
#         'cvt_lag5': uniform_filter1d(cvt, size=5, mode='nearest'),
#         'cvt_lag22': uniform_filter1d(cvt, size=22, mode='nearest')
#     })
#     model_data = model_data.dropna()
#     return model_data
#
#
#
#
# # Create model matrix
# def create_model_matrix(data, r2t, cjt, cvt):
#     model_data = pd.DataFrame({
#         'RV': data['RV'],
#         'r2t_lag1': np.roll(r2t, 1),
#         'r2t_lag5': uniform_filter1d(r2t, size=5, mode='nearest'),
#         'r2t_lag22': uniform_filter1d(r2t, size=22, mode='nearest'),
#         'cjt_lag1': np.roll(cjt, 1),
#         'cjt_lag5': uniform_filter1d(cjt, size=5, mode='nearest'),
#         'cjt_lag22': uniform_filter1d(cjt, size=22, mode='nearest'),
#         'cvt_lag1': np.roll(cvt, 1),
#         'cvt_lag5': uniform_filter1d(cvt, size=5, mode='nearest'),
#         'cvt_lag22': uniform_filter1d(cvt, size=22, mode='nearest')
#     })
#     model_data = model_data.dropna()
#     return model_data
#
#
# # Step 1: Loss function for optimizing lanta1, lanta2, lanta3
# def loss_function_lanta(params, data):
#     lanta1, lanta2, lanta3 = params
#
#     # Calculate measures
#     r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, data['returns'][:i + 1].values, lanta1) for i in
#            range(len(data))]
#     cjt = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['JV'][:i + 1].values, lanta2) for i in range(len(data))]
#     cvt = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['C_t'][:i + 1].values, lanta3) for i in range(len(data))]
#
#     # Create model matrix
#     model_data = create_model_matrix(data, r2t, cjt, cvt)
#     X = model_data.drop('RV', axis=1).values
#     y = model_data['RV'].values
#
#     # Perform cross-validation for LASSO
#     try:
#         lasso_model = LassoCV(cv=10).fit(X, y)
#         lambda_min = lasso_model.alpha_
#
#         # Predict and compute adjusted R-squared
#         y_pred = lasso_model.predict(X)
#         residuals = y - y_pred
#         tss = np.sum((y - np.mean(y)) ** 2)
#         rss = np.sum(residuals ** 2)
#         n = len(y)
#         p = np.sum(lasso_model.coef_ != 0)  # Non-zero coefficients
#         r_squared_adj = 1 - (rss / (n - p - 1)) / (tss / (n - 1))
#
#         return -r_squared_adj  # Return negative for minimization
#     except Exception as e:
#         print(f"Error in LASSO cross-validation: {e}")
#         return np.inf  # Return Inf for failed fits
#
#
# # Step 2: Optimize lanta1, lanta2, lanta3
# initial_params = np.random.uniform(1e-6, 20, 3)  # Initial values for lanta1, lanta2, lanta3
#
# # Perform optimization using L-BFGS-B method
# opt_result = minimize(
#     fun=loss_function_lanta,
#     x0=initial_params,
#     args=(data_get_cj,),
#     method='L-BFGS-B',
#     bounds=((1e-6, 20), (1e-6, 20), (1e-6, 20)),
#     options={'maxiter': 1000, 'disp': True}
# )
#
# # Extract optimized lanta parameters
# lanta1_opt, lanta2_opt, lanta3_opt = opt_result.x
#
#
# # Step 3: Final LASSO analysis with optimized lanta and cross-validated lambda
# def final_lasso_analysis(data, lanta1, lanta2, lanta3):
#     r2t = [conv_fun2_vectorized(TSPL_kernel_vectorized, data['returns'][:i + 1].values, lanta1) for i in
#            range(len(data))]
#     cjt = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['JV'][:i + 1].values, lanta2) for i in range(len(data))]
#     cvt = [conv_fun1_vectorized(TSPL_kernel_vectorized, data['C_t'][:i + 1].values, lanta3) for i in range(len(data))]
#
#     # Create model matrix
#     model_data = create_model_matrix(data, r2t, cjt, cvt)
#     X = model_data.drop('RV', axis=1).values
#     y = model_data['RV'].values
#
#     # Cross-validate and fit final LASSO model
#     lasso_model = LassoCV(cv=15).fit(X, y)
#     lambda_min = lasso_model.alpha_
#
#     # Extract coefficients and selected variables
#     coefficients = lasso_model.coef_
#     selected_vars = np.where(coefficients != 0)[0]
#     variable_names = model_data.drop('RV', axis=1).columns
#
#     # Calculate performance metrics
#     y_pred = lasso_model.predict(X)
#     mse = mean_squared_error(y, y_pred)
#     r_squared = r2_score(y, y_pred)
#
#     results = {
#         'coefficients': coefficients,
#         'selected_variables': variable_names[selected_vars],
#         'eliminated_variables': variable_names[~np.isin(np.arange(len(variable_names)), selected_vars)],
#         'performance': {
#             'mse': mse,
#             'r_squared': r_squared,
#             'lambda': lambda_min
#         },
#         'model': lasso_model
#     }
#
#     print("LASSO Regression Results:")
#     print("========================")
#     print("\nSelected Variables:")
#     print(results['selected_variables'])
#     print("\nModel Performance:")
#     print(f"R-squared: {r_squared:.4f}")
#     print(f"MSE: {mse:.4f}")
#     print(f"Lambda: {lambda_min:.6f}")
#
#     return results
#
#
# # Step 4: Analyze final results
# final_results = final_lasso_analysis(
#     data=data_get_cj,
#     lanta1=lanta1_opt,
#     lanta2=lanta2_opt,
#     lanta3=lanta3_opt
# )
#
#
# # Extract optimized parameters
# estimated_params = opt_result.x
# lanta1, lanta2, lanta3 = estimated_params
#
# # Print results
# print("Optimization Results:")
# print(f"lanta1: {lanta1:.6f}")
# print(f"lanta2: {lanta2:.6f}")
# print(f"lanta3: {lanta3:.6f}")
# print(f"Success: {opt_result.success}")
# print(f"Message: {opt_result.message}")

# lanta1=14.9866005
# lanta2=0.2619938
# lanta3= 14.6028617

# no lasso
lanta1= 9.52048
lanta2= 0.063294
lanta3 = 9.313669

# lasso
# lanta1=7.162356
# lanta2= 15.249771
# lanta3= 9.480584

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


model_data = pd.DataFrame({
    'RV': RV['RV'],
    'r2t_lag1': pd.Series(r2t).shift(1),
    'r2t_lag5': pd.Series(r2t).rolling(window=5, min_periods=1).mean().shift(1),
    'r2t_lag22': pd.Series(r2t).rolling(window=22, min_periods=1).mean().shift(1),
    'cjt_lag1': pd.Series(cjt).shift(1),
    'cjt_lag5': pd.Series(cjt).rolling(window=5, min_periods=1).mean().shift(1),
    'cjt_lag22': pd.Series(cjt).rolling(window=22, min_periods=1).mean().shift(1),
    'cvt_lag1': pd.Series(cvt).shift(1),
    'cvt_lag5': pd.Series(cvt).rolling(window=5, min_periods=1).mean().shift(1),
    'cvt_lag22': pd.Series(cvt).rolling(window=22, min_periods=1).mean().shift(1),
})

# Remove NA values
model_data = model_data.dropna()

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

df_predictions_lr.to_csv('cj_pd300.csv', index=False)
