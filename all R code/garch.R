# 加载必要的包
library(rugarch)
library(xts)
library(zoo)
library(TTR)
library(quantmod)
library(tidyverse)
library(hawkes)
library(highfrequency)
library(dplyr)

setwd("c:/Users/lenovo/Desktop/HAR")
df <- read.csv("data_idx.csv")
data_ret <- df %>%
  dplyr::select(time, code, close) %>%
  set_names(c("DT", "id", "PRICE")) %>%
  na.omit() %>%
  dplyr::group_by(id) %>%
  dplyr::group_split() %>%
  lapply(., function(x) data.frame(DT = x$DT, id = x$id, Ret = makeReturns(x$PRICE))) %>%
  bind_rows()

# 按组进行分类
group_summary <- data_ret %>% 
  group_by(id) %>%
  summarise(NumObservations = n())

# 建立一个可以用于HAR的数据框，提取第一集合
data_filtered <- data_ret %>%
  filter(id == "000001.XSHG")
data_filtered <- data_filtered[,-4]

# 在提取第一个id 以后，我们需要将return 值去计算每日RV值
data_filtered$DT <- as.Date(data_filtered$DT)

# 计算每日RV值
RV <- data_filtered %>%
  group_by(DT) %>%
  summarise(RV = sum(Ret^2))


RV$DT <- as.Date(RV$DT)

# 数据拆分
split_index <- floor(0.8 * nrow(RV))
train_data <- RV[1:split_index, ]
test_data <- RV[(split_index + 1):nrow(RV), ]

# 将完整的数据转换为 xts 对象
rv_xts <- xts(RV$RV, order.by = RV$DT)
rv_xts <- na.omit(rv_xts)

# 将测试数据转换为 xts 对象
rv_test_xts <- xts(test_data$RV, order.by = test_data$DT)

# 定义 GARCH(1,1) 模型规格
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
  distribution.model = "std"  # 使用 t 分布
)

# 存储预测值的向量
predictions <- numeric(length = nrow(test_data))

# 存储预测对应的日期
prediction_dates <- test_data$DT

# 设置计时器开始
start_time <- Sys.time()

# 进行滚动预测
for (i in 1:nrow(test_data)) {
  # 当前的训练数据，包括原始训练集和测试集中的前 i-1 个数据
  train_indices <- 1:(split_index + i - 1)
  train_subset <- rv_xts[train_indices]
  
  # 拟合模型
  fit <- ugarchfit(spec = spec, data = train_subset, solver = 'hybrid', solver.control = list(trace = 0), silent = TRUE)
  
  # 检查模型是否成功收敛
  if (fit@fit$convergence == 0) {
    # 进行一步预测
    forecast <- ugarchforecast(fit, n.ahead = 1)
    
    # 提取预测的条件标准差（波动率）
    sigma_forecast <- as.numeric(sigma(forecast))
    
    # 存储预测结果
    predictions[i] <- sigma_forecast
    
  } else {
    warning(paste("模型在滚动步数", i, "未能收敛。"))
    predictions[i] <- NA  # 如果模型未收敛，存储 NA
  }
  
  # 可选：显示进度
  if (i %% 50 == 0) {
    cat("已完成", i, "步，共", nrow(test_data), "步。\n")
  }
}

# 计算并显示总耗时
end_time <- Sys.time()
total_time <- end_time - start_time
cat("滚动预测总耗时：", total_time, "\n")

# 构建预测结果数据框
prediction_df <- data.frame(
  Date = prediction_dates,
  PredictedSigma = predictions,
  ActualRV = test_data$RV
)

# 查看前几行预测结果
head(prediction_df)

# 计算预测误差
prediction_df$Error <- prediction_df$ActualRV - prediction_df$PredictedSigma

# 计算均方误差（MSE）
mse <- mean(prediction_df$Error^2, na.rm = TRUE)
cat("均方误差（MSE）：", mse, "\n")




#-------- EGARCH GJR-GARCH TARCH ----------------

# EGARCH(1,1)
spec_egarch <- ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
  distribution.model = "std"
)

# GJR-GARCH(1,1)
spec_gjr <- ugarchspec(
  variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
  distribution.model = "std"
)

# TARCH(1,1)
spec_tarch <- ugarchspec(
  variance.model = list(model = "fGARCH", submodel = "TGARCH", garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
  distribution.model = "std"
)

# 定义滚动预测函数
rolling_forecast <- function(spec, rv_xts, split_index, test_data) {
  predictions <- numeric(length = nrow(test_data))
  prediction_dates <- test_data$DT
  
  for (i in 1:nrow(test_data)) {
    train_indices <- 1:(split_index + i - 1)
    train_subset <- rv_xts[train_indices]
    
    fit <- ugarchfit(spec = spec, data = train_subset, solver = 'hybrid', solver.control = list(trace = 0), silent = TRUE)
    
    if (fit@fit$convergence == 0) {
      forecast <- ugarchforecast(fit, n.ahead = 1)
      sigma_forecast <- as.numeric(sigma(forecast))
      predictions[i] <- sigma_forecast
    } else {
      warning(paste("模型在滚动步数", i, "未能收敛。"))
      predictions[i] <- NA
    }
    
    if (i %% 50 == 0) {
      cat("已完成", i, "步，共", nrow(test_data), "步。\n")
    }
  }
  
  prediction_df <- data.frame(
    Date = prediction_dates,
    PredictedSigma = predictions,
    ActualRV = test_data$RV
  )
  
  return(prediction_df)
}

# 进行滚动预测

# EGARCH
cat("正在使用 EGARCH 模型进行滚动预测...\n")
forecast_egarch <- rolling_forecast(spec_egarch, rv_xts, split_index, test_data)

# GJR-GARCH
cat("正在使用 GJR-GARCH 模型进行滚动预测...\n")
forecast_gjr <- rolling_forecast(spec_gjr, rv_xts, split_index, test_data)



#--------------- TGARCH -------------------
# 定义 TARCH(1,1) 模型规格
spec_tarch <- ugarchspec(
  variance.model = list(
    model = "fGARCH",
    submodel = "TGARCH",
    garchOrder = c(1, 1),
    include.sumgarch = FALSE  # 添加此参数
  ),
  mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
  distribution.model = "std"
)

# 定义滚动预测函数，加入 fit.control 中的 robust 参数，并使用 tryCatch
rolling_forecast <- function(spec, rv_xts, split_index, test_data) {
  predictions <- numeric(length = nrow(test_data))
  prediction_dates <- test_data$DT
  
  for (i in 1:nrow(test_data)) {
    train_indices <- 1:(split_index + i - 1)
    train_subset <- rv_xts[train_indices]
    
    fit_result <- tryCatch({
      fit <- ugarchfit(
        spec = spec,
        data = train_subset,
        solver = 'hybrid',
        solver.control = list(trace = 0),
        fit.control = list(robust = FALSE),  # 加入此行
        silent = TRUE
      )
      
      if (fit@fit$convergence == 0) {
        forecast <- ugarchforecast(fit, n.ahead = 1)
        sigma_forecast <- as.numeric(sigma(forecast))
        sigma_forecast
      } else {
        warning(paste("模型在滚动步数", i, "未能收敛。"))
        NA
      }
    }, error = function(e) {
      warning(paste("在滚动步数", i, "发生错误：", e$message))
      NA
    })
    
    predictions[i] <- fit_result
    
    if (i %% 50 == 0) {
      cat("已完成", i, "步，共", nrow(test_data), "步。\n")
    }
  }
  
  prediction_df <- data.frame(
    Date = prediction_dates,
    PredictedSigma = predictions,
    ActualRV = test_data$RV
  )
  
  return(prediction_df)
}

# 运行 TARCH 模型的滚动预测
cat("正在使用 TARCH 模型进行滚动预测...\n")
forecast_tarch <- rolling_forecast(spec_tarch, rv_xts, split_index, test_data)



#---------------------- FGARCH -------------------

# 定义 fGARCH(1,1) 模型规格
spec_fgarch <- ugarchspec(
  variance.model = list(
    model = "fGARCH", # 指定模型为fGARCH
    garchOrder = c(1, 1), # 指定GARCH的阶数为(1,1)
    submodel = "GARCH", # 指定子模型为GARCH
    external.regressors = NULL, # 如果有外生变量则在这里指定
    variance.targeting = FALSE # 是否进行方差盯住，通常设置为FALSE
  ),
  mean.model = list(armaOrder = c(0, 0), include.mean = FALSE), # 均值模型为常数，这里设置为FALSE
  distribution.model = "std" # 指定误差项的分布为学生t分布
)

# 定义滚动预测函数
rolling_forecast <- function(spec, rv_xts, split_index, test_data) {
  predictions <- numeric(length = nrow(test_data))
  prediction_dates <- test_data$DT
  
  for (i in 1:nrow(test_data)) {
    train_indices <- 1:(split_index + i - 1)
    train_subset <- rv_xts[train_indices]
    
    fit_result <- tryCatch({
      fit <- ugarchfit(
        spec = spec,
        data = train_subset,
        solver = 'hybrid',
        solver.control = list(trace = 0),
        fit.control = list(robust = FALSE),  # 使用稳健拟合选项
        silent = TRUE
      )
      
      if (fit@fit$convergence == 0) {
        forecast <- ugarchforecast(fit, n.ahead = 1)
        sigma_forecast <- as.numeric(sigma(forecast))
        sigma_forecast
      } else {
        warning(paste("模型在滚动步数", i, "未能收敛。"))
        NA
      }
    }, error = function(e) {
      warning(paste("在滚动步数", i, "发生错误：", e$message))
      NA
    })
    
    predictions[i] <- fit_result
    
    if (i %% 50 == 0) {
      cat("已完成", i, "步，共", nrow(test_data), "步。\n")
    }
  }
  
  prediction_df <- data.frame(
    Date = prediction_dates,
    PredictedSigma = predictions,
    ActualRV = test_data$RV
  )
  
  return(prediction_df)
}

# 运行 fGARCH 模型的滚动预测
cat("正在使用 fGARCH 模型进行滚动预测...\n")
forecast_fgarch <- rolling_forecast(spec_fgarch, rv_xts, split_index, test_data)




#------------- ARFIMA-GARCH ----------------

# 定义 ARFIMA(1, d, 1)-GARCH(1,1) 模型规格
spec_arfima <- ugarchspec(
  variance.model = list(
    model = "sGARCH",
    garchOrder = c(1, 1)
  ),
  mean.model = list(
    armaOrder = c(1, 1),
    include.mean = TRUE,
    arfima = TRUE
  ),
  distribution.model = "std"
)

# 拟合模型
fit_arfima <- ugarchfit(
  spec = spec_arfima,
  data = rv_train_xts,
  solver = 'hybrid'
)

# 检查模型是否成功收敛
if (fit_arfima@fit$convergence == 0) {
  cat("ARFIMA-GARCH 模型成功收敛。\n")
} else {
  cat("ARFIMA-GARCH 模型未能收敛，可能存在问题。\n")
}

# 查看模型参数估计值
params <- coef(fit_arfima)
print("ARFIMA-GARCH 模型参数估计值：")
print(params)

# 定义滚动预测函数
rolling_forecast_arfima <- function(spec, rv_xts, split_index, test_data) {
  predictions <- numeric(length = nrow(test_data))
  prediction_dates <- test_data$DT
  
  for (i in 1:nrow(test_data)) {
    train_indices <- 1:(split_index + i - 1)
    train_subset <- rv_xts[train_indices]
    
    fit_result <- tryCatch({
      fit <- ugarchfit(
        spec = spec,
        data = train_subset,
        solver = 'hybrid',
        solver.control = list(trace = 0),
        fit.control = list(robust = FALSE),
        silent = TRUE
      )
      
      if (fit@fit$convergence == 0) {
        forecast <- ugarchforecast(fit, n.ahead = 1)
        sigma_forecast <- as.numeric(sigma(forecast))
        sigma_forecast
      } else {
        warning(paste("模型在滚动步数", i, "未能收敛。"))
        NA
      }
    }, error = function(e) {
      warning(paste("在滚动步数", i, "发生错误：", e$message))
      NA
    })
    
    predictions[i] <- fit_result
    
    if (i %% 50 == 0) {
      cat("已完成", i, "步，共", nrow(test_data), "步。\n")
    }
  }
  
  prediction_df <- data.frame(
    Date = prediction_dates,
    PredictedSigma = predictions,
    ActualRV = test_data$RV
  )
  
  return(prediction_df)
}

# 进行滚动预测
cat("正在使用 ARFIMA-GARCH 模型进行滚动预测...\n")
forecast_arfima <- rolling_forecast_arfima(spec_arfima, rv_xts, split_index, test_data)

# 计算预测误差
forecast_arfima$Error <- forecast_arfima$ActualRV - forecast_arfima$PredictedSigma

# 计算均方误差（MSE）
mse_arfima <- mean(forecast_arfima$Error^2, na.rm = TRUE)
cat("ARFIMA-GARCH 模型的均方误差（MSE）：", mse_arfima, "\n")




#---------------MS GARCH -------------------
library(MSGARCH)
library(xts)

# 转换日期格式
RV$DT <- as.Date(RV$DT)

# 数据拆分
split_index <- floor(0.8 * nrow(RV))
train_data <- RV[1:split_index, ]
test_data <- RV[(split_index + 1):nrow(RV), ]

# 检查并移除缺失值
rv_xts <- na.omit(xts(RV$RV, order.by = RV$DT))
rv_train_xts <- na.omit(xts(train_data$RV, order.by = train_data$DT))
rv_test_xts <- na.omit(xts(test_data$RV, order.by = test_data$DT))

rolling_forecast_msgarch <- function(spec, rv_xts, split_index, test_data) {
  predictions <- numeric(nrow(test_data))
  prediction_dates <- index(test_data)
  
  for (i in 1:nrow(test_data)) {
    # 提取训练数据
    train_indices <- 1:(split_index + i - 1)
    train_subset <- as.numeric(rv_xts[train_indices])
    
    # 使用新的训练数据拟合 MSGARCH 模型
    fit_result <- FitML(spec = spec, data = train_subset)
    
    # 使用预测函数进行滚动预测
    forecast <- predict(fit_result, nahead = 1)
    
    # 提取预测的条件波动率
    predictions[i] <- forecast$vol[1]  # 提取第一个预测值
    
    # 打印进度
    if (i %% 50 == 0) {
      cat("已完成", i, "步，共", nrow(test_data), "步。\n")
    }
  }
  
  # 生成预测结果的数据框
  prediction_df <- data.frame(
    Date = prediction_dates,
    PredictedRV = predictions,
    ActualRV = as.numeric(test_data)
  )
  
  return(prediction_df)
}

# 创建 MS-GARCH 模型规格（两状态）
ms_spec <- CreateSpec(
  variance.spec = list(model = c("sGARCH")),
  distribution.spec = list(distribution = c("std")),
  switch.spec = list(do.mix = FALSE)
)

# 初始拟合
ms_fit <- FitML(spec = ms_spec, data = as.numeric(rv_train_xts))

# 检查初始模型拟合情况
summary(ms_fit)

# 进行滚动预测
cat("正在使用 MS-GARCH 模型进行滚动预测...\n")
forecast_msgarch <- rolling_forecast_msgarch(ms_spec, rv_xts, split_index, rv_test_xts)

# 计算预测误差
forecast_msgarch$Error <- forecast_msgarch$ActualRV - forecast_msgarch$PredictedRV
mse_msgarch <- mean(forecast_msgarch$Error^2, na.rm = TRUE)
cat("MS-GARCH 模型的均方误差（MSE）：", mse_msgarch, "\n")




#---------------GAS GARCH-------------------
library(GAS)

rolling_forecast_gas <- function(spec, rv_xts, split_index, test_data) {
  predictions <- numeric(nrow(test_data))
  prediction_dates <- index(test_data)
  
  for (i in 1:nrow(test_data)) {
    # 提取训练数据
    train_indices <- 1:(split_index + i - 1)
    train_subset <- as.numeric(rv_xts[train_indices])
    
    # 重新拟合 GAS 模型
    fit_result <- FitML(spec = spec, data = train_subset)
    
    # 检查是否成功拟合并提取预测值
    forecast <- predict(fit_result, nahead = 1)
    if (!is.null(forecast)) {
      predictions[i] <- forecast$vol[1]  # 提取条件波动率
    } else {
      predictions[i] <- NA
      warning(paste("模型在滚动步数", i, "未能预测。"))
    }
    
    # 打印进度
    if (i %% 50 == 0) {
      cat("已完成", i, "步，共", nrow(test_data), "步。\n")
    }
  }
  
  # 返回预测结果数据框
  prediction_df <- data.frame(
    Date = prediction_dates,
    PredictedRV = predictions,
    ActualRV = as.numeric(test_data)
  )
  
  return(prediction_df)
}

gas_spec <- CreateSpec(
  variance.spec = list(model = c("sGARCH")),
  distribution.spec = list(distribution = c("std"))
)

# 初始拟合
gas_fit <- FitML(spec = gas_spec, data = as.numeric(rv_train_xts))

# 检查初始模型拟合结果
summary(gas_fit)

# 进行滚动预测
cat("正在使用 GAS-GARCH 模型进行滚动预测...\n")
forecast_gas <- rolling_forecast_gas(gas_spec, rv_xts, split_index, rv_test_xts)

# 计算预测误差
forecast_gas$Error <- forecast_gas$ActualRV - forecast_gas$PredictedRV
mse_gas <- mean(forecast_gas$Error^2, na.rm = TRUE)
cat("GAS-GARCH 模型的均方误差（MSE）：", mse_gas, "\n")

