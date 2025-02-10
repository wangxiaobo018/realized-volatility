
#------------------------------ HAR-RS 模型
setwd("c:/Users/lenovo/Desktop/HAR")
df <- read.csv("day_data.csv")
# 注意在计算R1,t 的r1,t 的计算和在HAR模型中计算是不一样的，注意这个
#Sti表示的是过去每日收益的回报也就是RETURNs
library(tidyverse)
library(hawkes)
library(highfrequency)
library(dplyr)
library(zoo)
data_ret <- df %>% 
 select(time, code, close) %>%
 set_names(c("DT", "id", "PRICE")) %>% 
 na.omit() %>% 
 group_split(id) %>%
 lapply(., function (x) data.frame(DT = x$DT, id=x$id,Ret = makeReturns(x$PRICE)))%>%
 bind_rows()

# 按组进行分类
group_summary <- data_ret %>% 
 group_by(id) %>%
 summarise(NumObservations = n())


# 进行分类并提取第一类数据
data_filtered <- df %>%
 select(time, code, close) %>%
 set_names(c("DT", "id", "price")) %>%
 na.omit() %>%
 filter(id == "000001.XSHG")


ret <- function(prices) {
 n <- length(prices)
 returns <- rep(NA, n)
 for (i in 2:n) {
  if(prices[i-1]==0){
   returns[i] <- NA
  }else{
   returns[i] <- (prices[i] - prices[i-1]) / prices[i-1]
  }
 }
 returns[1] <-0
 return(returns)
}
returns <- ret(data_filtered$price)
returns <- as.data.frame(returns)



#-------------------------------计算har-rs 等若干模型


setwd("c:/Users/lenovo/Desktop/HAR")
df <- read.csv("data_idx.csv")
data_ret <- df %>% 
 select(time, code, close) %>%
 set_names(c("DT", "id", "PRICE")) %>% 
 na.omit() %>% 
 group_split(id) %>%
 lapply(., function (x) data.frame(DT = x$DT, id=x$id,Ret = makeReturns(x$PRICE)))%>%
 bind_rows()

# 按组进行分类
group_summary <- data_ret %>% 
 group_by(id) %>%
 summarise(NumObservations = n())
print(group_summary)
# 建立一个可以用于HAR的数据框，提取第一集合
data_cj<- data_ret %>%
 filter(id == "000001.XSHG")
data_cj <- data_cj[,-4]




#-----------------------  以及先写成函数，只要将计算出returns的正负区分开，再进行求RS+和RS-
calculate_RS <- function(data) {
 # 分离正收益和负收益
 data$Positive_Returns <- ifelse(data$Ret > 0, data$Ret, 0)
 data$Negative_Returns <- ifelse(data$Ret < 0, data$Ret, 0)
 
 RS_plus <- sum(data$Positive_Returns^2, na.rm = TRUE)
 RS_minus <- sum(data$Negative_Returns^2, na.rm = TRUE)
 date <- data$DT[1]
 return(data.frame(Date = date, RS_plus = RS_plus, RS_minus = RS_minus))
}

#--------------------- 将其应用到所有的天就得出了RS+ RS-这两个值
result <- data_cj %>%
 group_by(Date = as.Date(DT)) %>%
 do(calculate_RS(.))
result <- ungroup(result)
result$RS_plus <- as.numeric(as.character(result$RS_plus))
result$RS_minus <- as.numeric(as.character(result$RS_minus))
data <- cbind(result[,-1],returns)



#--------------------  上面是计算完了returns 还有 RS+ RS- ，下面再计算RV的值
setwd("c:/Users/lenovo/Desktop/HAR")
df <- read.csv("data_idx.csv")
data_ret <- df %>% 
 select(time, code, close) %>%
 set_names(c("DT", "id", "PRICE")) %>% 
 na.omit() %>% 
 group_split(id) %>%
 lapply(., function (x) data.frame(DT = x$DT, id=x$id,Ret = makeReturns(x$PRICE)))%>%
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
RV <- data_filtered %>%
 group_by(DT) %>%
 summarise(RV = (sum(Ret^2)))

data_har_cj <- cbind(RV,data)
data <- data_har_cj[,-1]


#---------------------  下面是计算har-rs模型
library(zoo)

rsp_lag1 <- lag(data$RS_plus, 1)
rsp_lag5 <- rollmean(data$RS_plus, 5, align = "right", fill = NA)
rsp_lag22 <- rollmean(data$RS_plus, 22, align = "right", fill = NA)
rsm_lag1 <- lag(data$RS_minus, 1)
rsm_lag5 <- rollmean(data$RS_minus, 5, align = "right", fill = NA)
rsm_lag22 <- rollmean(data$RS_minus, 22, align = "right", fill = NA)

# 创建数据框
model_data <- data.frame(rsp_lag1, rsp_lag5, rsp_lag22, rsm_lag1, rsm_lag5, rsm_lag22, RV = data$RV)
model_data <- na.omit(model_data)
model <- lm(RV~., data = model_data)
model_aic <- AIC(model)
model_bic <- BIC(model)
print(model_aic)
print(model_bic)



test_size <- 1000
train_data <- model_data[1:(nrow(model_data) - test_size), ]
test_data <- model_data[(nrow(model_data) - test_size + 1):nrow(model_data), ]


predictionrs1 <- numeric(nrow(test_data))
predictionrs5 <- numeric(nrow(test_data))
predictionrs22 <- numeric(nrow(test_data))
actual_ree1 <- numeric(nrow(test_data))
actual_ree5 <- numeric(nrow(test_data))
actual_ree22 <- numeric(nrow(test_data))
for(i in 1:nrow(test_data)){
  train_start <- i
  train_end <- i + nrow(train_data) - 1
  train_windows <- model_data[train_start:train_end, ]
  model <- lm(RV ~ ., data = train_windows)
  predictionrs1[i] <- predict(model, newdata = model_data[train_end + 1, ])
  predictionrs5[i] <- predict(model, newdata = model_data[train_end + 5, ])
  predictionrs22[i] <- predict(model, newdata = model_data[train_end + 22, ])
  actual_ree1[i] <- model_data[train_end+1,1]
  actual_ree5[i] <- model_data[train_end+5,1]
  actual_ree22[i] <- model_data[train_end+22,1]
}
print(sum(predictionrs1<0))
rs300 <- cbind(actual_ree1,actual_ree5,actual_ree22, predictionrs1, predictionrs5, predictionrs22)
rs300 <- cbind(ifelse(actual_ree1 < 0, 0.00001, actual_ree1),
               ifelse(actual_ree5 < 0, 0.00001, actual_ree5),
               ifelse(actual_ree22 < 0, 0.00001, actual_ree22),
               ifelse(predictionrs1 < 0, 0.00001, predictionrs1),
               ifelse(predictionrs5 < 0, 0.00001, predictionrs5),
               ifelse(predictionrs22 < 0, 0.00001, predictionrs22))
write.csv(rs300,"HARRS800.csv")


test_size <- 300
train_data <- model_data[1:(nrow(model_data) - test_size), ]
test_data <- model_data[(nrow(model_data) - test_size + 1):nrow(model_data), ]


predictionrs600 <- numeric(nrow(test_data))
for(i in 1:nrow(test_data)){
  train_start <- i
  train_end <- i + nrow(train_data) - 1
  train_windows <- model_data[train_start:train_end, ]
  model <- lm(RV ~ ., data = train_windows)
  predictionrs600[i] <- predict(model, newdata = model_data[train_end + 1, ])
}
write.csv(predictionrs600,"HARRS600.csv")


spilt_index <- floor(0.8 * nrow(data))
data_rs_train <- data[1:spilt_index,]
data_rs_test <- data[(spilt_index+1):nrow(data),]


library(dplyr)
library(zoo)
data_rs <- data_rs_train %>%
  mutate(
    rs_t_lag1 = lag(RS_plus, 1),
    rs_t_lag5 = rollmean(RS_plus, 5, align = "right", fill = NA),
    rs_t_lag22 = rollmean(RS_plus, 22, align = "right", fill = NA),
    rs_f_lag1 = lag(RS_minus, 1),
    rs_f_lag5 = rollmean(RS_minus, 5, align = "right", fill = NA),
    rs_f_lag22 = rollmean(RS_minus, 22, align = "right", fill = NA)
  )

# 移除包含NA的行
data_rs <- na.omit(data_rs)

# 建立线性模型
rs_model <- lm(RV ~ rs_t_lag1 + rs_t_lag5 + rs_t_lag22 + rs_f_lag1 + rs_f_lag5 + rs_f_lag22, data = data_rs)

# 显示模型摘要
summary(rs_model)
coe <- summary(rs_model)$coef
cof <- coe[,1]
windows(1,5,22)






#--------------------- 以下是计算的HAR-PD-RS的

TSPL_kernel_vectorized <- function(diff_times, lanta) {
  lanta * exp(-lanta * diff_times)
}

conv_fun1_vectorized <- function(kernel, x, lanta) {
  diff_times <- seq_along(x) - 1
  weights <- rev(kernel(diff_times, lanta))
  normalized_weights <- weights / sum(weights)  # 归一化权重
  sum(normalized_weights * x)
}


conv_fun2_vectorized <- function(kernel, x, lanta) {
  diff_times <- seq_along(x) - 1
  weights <- rev(kernel(diff_times, lanta))
  normalized_weights <- weights / sum(weights)  # 归一化权重
  sum(normalized_weights * x^2)
}



loss_function_vectorized <- function(params, data) {
  lanta1 <- params[1]
  lanta2 <- params[2]
  beta <- params[3:12]

  R1t <- sapply(seq_along(data$returns), function(i) conv_fun1_vectorized(TSPL_kernel_vectorized, data$returns[1:i], lanta1))
  RS_p <- sapply(seq_along(data$RS_plus), function(i) conv_fun2_vectorized(TSPL_kernel_vectorized, data$RS_plus[1:i], lanta2))
  RS_m <- sapply(seq_along(data$RS_minus), function(i) conv_fun2_vectorized(TSPL_kernel_vectorized, data$RS_minus[1:i], lanta2))
  data_df <- data.frame(RV = as.numeric(data$RV), 
                        R1t = as.numeric((R1t)), 
                        RS_p = as.numeric(sqrt(RS_p)), 
                        RS_m = as.numeric(sqrt(RS_m)))
  
  R1t_1 <- lag(data_df$R1t,1)
  R1t_5 <- rollmean(data_df$R1t,5,align="right",fill=NA)
  R1t_22 <- rollmean(data_df$R1t,22,align="right",fill=NA)
  RS_p_1 <- lag(data_df$RS_p,1)
  RS_p_5 <- rollmean(data_df$RS_p,5,align="right",fill=NA)
  RS_p_22 <- rollmean(data_df$RS_p,22,align="right",fill=NA)
  RS_m_1 <- lag(data_df$RS_m,1)
  RS_m_5 <- rollmean(data_df$RS_m,5,align="right",fill=NA)
  RS_m_22 <- rollmean(data_df$RS_m,22,align="right",fill=NA)
  
  # 添加滞后项
  data_df <- cbind(data_df, R1t_1, R1t_5, R1t_22)
  data_df <- cbind(data_df, RS_p_1, RS_p_5, RS_p_22)
  data_df <- cbind(data_df, RS_m_1, RS_m_5, RS_m_22)
  
  # 设置新列名
  names(data_df)[5:ncol(data_df)] <- c("R1t_1", "R1t_5", "R1t_22", "RS_p_1", "RS_p_5", "RS_p_22", "RS_m_1", "RS_m_5", "RS_m_22")
  
  # 只选取滞后项列
  predictors <- c("R1t_1", "R1t_5", "R1t_22", "RS_p_1", "RS_p_5", "RS_p_22", "RS_m_1", "RS_m_5", "RS_m_22")
  pred_RV <- beta[1] + as.matrix(data_df[, predictors]) %*% beta[-1]
  
  mse <- mean((data_df$RV - pred_RV)^2, na.rm = TRUE)
  return(mse)
} 
initial_params <- runif(12)
opt_result_rs <- optim(initial_params, loss_function_vectorized, data = data_har_cj,
                       method = "L-BFGS-B",
                       lower = c(1e-6, rep(-2, 11)),
                       upper = c(rep(2, 12)),
                       control = list(maxit = 1000, trace = 1))

# 输出结果
print(opt_result_rs$par)
print(opt_result_rs$value)

estimated_params <- opt_result_rs$par[1:11]
lanta1 <- estimated_params[1]
lanta2 <- estimated_params[2]


# 使用管道操作符简化计算 R1t 的值
R1t <- NULL
for(i in seq(data_har_cj$returns)){
 R1t[i] <- conv_fun1_vectorized(TSPL_kernel_vectorized, data_har_cj$returns[1:i],lanta1 )
}
RS_plus <- NULL
for(i in seq(data_har_cj$RS_plus)){
 RS_plus[i] <- conv_fun2_vectorized(TSPL_kernel_vectorized, data_har_cj$RS_plus[1:i],lanta2 )
}

RS_minus <- NULL
for(i in seq(data_har_cj$RS_minus)){
 RS_minus[i] <- conv_fun2_vectorized(TSPL_kernel_vectorized, data_har_cj$RS_minus[1:i],lanta2 )
}


r1t <- lag(R2t, 1)
r1t_lag5 <- rollmean(R2t, 5, align = "right", fill = NA)
r1t_lag22 <- rollmean(R2t, 22, align = "right", fill = NA)
rs_p_lag1 <- lag(RS_plus, 1)
rs_p_lag5 <- rollmean(RS_plus, 5, align = "right", fill = NA)
rs_p_lag22 <- rollmean(RS_plus, 22, align = "right", fill = NA)
rs_m_lag1 <- lag(RS_minus, 1)
rs_m_lag5 <- rollmean(RS_minus, 5, align = "right", fill = NA)
rs_m_lag22 <- rollmean(RS_minus, 22, align = "right", fill = NA)


data_rs <- data.frame(rs_p_lag1 , rs_p_lag5 , rs_p_lag22 ,
                      rs_m_lag1 , rs_m_lag5 , rs_m_lag22 ,
                      r1t , r1t_lag5 , r1t_lag22,RV = data_har_cj$RV)
data_rs <- na.omit(data_rs)
# 划分训练集和测试集
split_index <- floor(0.8 * nrow(data_rs))
train_data_rs <- as.data.frame(data_rs[1:split_index, ])
test_data_rs <- as.data.frame(data_rs[(split_index + 1):nrow(data_rs), ])

u <- lm(RV ~ ., data = train_data_rs)
summary(u)





train_df <- data_rs[1:(nrow(data_rs)-600), ]
test_df <- data_rs[(nrow(data_rs)-599):nrow(data_rs), ]
last_600 <- numeric(nrow(test_df))

for(i in 1:nrow(test_df)){
  train_start <- i
  train_end <- i + nrow(train_df) - 1
  train_windows <- data[train_start:train_end, ]
  model <- lm(RV ~ ., data = train_windows)
  last_600[i] <- predict(model, newdata = data_rs[train_end + 1, ])
}
out_r2t <- data.frame(RV = test_df$RV, RV_pred = last_600) 
write.csv(out_r2t,"rspd600.csv")



predict_rspd1 <- numeric(nrow(test_data_rs))
for(i in 1:nrow(test_data_rs)){
  train_start <- i
  train_end <- i + nrow(train_data_rs) - 1
  train_windows <- data_rs[train_start:train_end, ]
  model <- lm(RV ~ ., data = train_windows)
  predict_rspd1[i] <- predict(model, newdata = data_rs[train_end + 1, ])
}
write.csv(predict_rspd1,"predict_rspd1.csv")

colnames(train_data_rs) <- c("RS_plus", "RS_minus", "RV", "R1t")
colnames(test_data_rs) <- c("RS_plus", "RS_minus", "RV", "R1t")




train_df <- data_rs[1:(nrow(data_rs)-1200), ]
test_df <- data_rs[(nrow(data_rs)-1199):nrow(data_rs), ]
last_600 <- numeric(nrow(test_df))

for(i in 1:nrow(test_df)){
  train_start <- i
  train_end <- i + nrow(train_df) - 1
  train_windows <- data_rs[train_start:train_end, ]
  model <- lm(RV ~ ., data = train_windows)
  last_600[i] <- predict(model, newdata = data_rs[train_end + 1, ])
}
out_r2t <-  last_600
write.csv(out_r2t,"newrs1200.csv")

