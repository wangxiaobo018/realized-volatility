
setwd("c:/Users/lenovo/Desktop/HAR")
data <- read.csv("DATA.csv")
data<- na.omit(data)
RV_true <- data$RV

har_models <- data[, grep("HAR", names(data))]
# #-----------检查data是否有NA
# sum(data <0 )
# # 检查是否有NA
# # 检查har_models中是否有0或负值
# if(any(har_models <= 0)) {
#   cat("There are non-positive values in the har_models which will cause issues with log and division.\n")
# }
# 
# # 检查最终的ql_results和mse_results是否有NA或Inf
# if(any(is.na(ql_results) | is.infinite(ql_results))) {
#   cat("There are NA or Inf values in ql_results.\n")
# }
# if(any(is.na(mse_results) | is.infinite(mse_results))) {
#   cat("There are NA or Inf values in mse_results.\n")
# }
# 
# 
# # 假设你的数据集名为 data
# # data <- read.csv("your_data.csv") # 假设这是你如何加载数据的方式
# 
# # 初始化一个列表来存储包含负数的列和对应的行索引
# negative_details <- list()
# 
# # 检查每一列
# for (col in names(data)) {
#   # 找到该列中负数的行索引
#   negative_indices <- which(data[[col]] < 0)
#   
#   # 如果存在负数，保存列名和对应的行索引
#   if (length(negative_indices) > 0) {
#     negative_details[[col]] <- negative_indices
#   }
# }
# 
# # 输出所有包含负数的列和行索引
# if (length(negative_details) > 0) {
#   print("Columns and rows with negative values:")
#   print(negative_details)
# } else {
#   print("No negative values found in the dataset.")
# }
# # # 计算每个模型的QLIKE和MSE损失,不求平均
ql_results <- sapply(har_models, function(x) (log(x) + RV_true / x))
mse_results <- sapply(har_models, function(x) (RV_true - x)^2)


# 去除异常值
ql_threshold <- quantile(ql_results, 0.99, na.rm = TRUE)
ql_results[ql_results > ql_threshold] <- ql_threshold

mse_threshold <- quantile(mse_results, 0.99, na.rm = TRUE)
mse_results[mse_results > mse_threshold] <- mse_threshold

# 引入MCS库
library(MCS)

# 每个模型的损失向量需要转为时间序列对象
ql_ts <- ts(ql_results)
mse_ts <- ts(mse_results)

set.seed(123)
mcs_ql <- MCSprocedure(ql_results, alpha = 0.25, B = 5000)
set.seed(456)
mcs_mse <- MCSprocedure(mse_results, alpha = 0.25, B= 5000)

set.seed(234)
mcs_ql_ts <- MCSprocedure(ql_ts, alpha = 0.1, B = 5000)
set.seed(567)
mcs_mse_ts <- MCSprocedure(mse_ts, alpha = 0.1, B = 5000)
