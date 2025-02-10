

library(xts)
library(zoo)
library(TTR)
library(quantmod)
library(tidyverse)
library(hawkes)
library(highfrequency)
library(dplyr)
library(lubridate)
setwd("c:/Users/lenovo/Desktop/HAR")
df <- read.csv("data_idx.csv")

# 按组进行分类
group_summary <- df %>%
  group_by(code) %>%
  summarise(NumObservations = n())

# 提取 code 为 "000001.XSHG" 的部分
data_filtered <- df %>%
  filter(code == "000001.XSHG")




get_re1 <- function(data,alpha){
  result <- data %>% 
    mutate(
      time_str = as.character(.data$time),
      date_str = substr(time_str, 1, 10),
      day = as.Date(date_str)
    ) %>% 
    filter(!is.na(day)) %>% 
    group_by(day) %>%
    mutate(
      Ret = (log(close) - log(dplyr::lag(close))
      )) %>%
    na.omit() %>%
    summarise(
      sigma = sd(Ret, na.rm = TRUE),
      emprical_plus = quantile(Ret, 1 - alpha),
      emprical_minus = quantile(Ret, alpha),
      rex_minus = sum(ifelse(Ret <= emprical_minus, Ret^2, 0), na.rm = TRUE),
      rex_plus = sum(ifelse(Ret >= emprical_plus, Ret^2, 0), na.rm = TRUE),
      rex_moderate = sum(ifelse(Ret > emprical_minus & Ret < emprical_plus, Ret^2, 0), na.rm = TRUE)
    ) %>%
    select(day,rex_minus,rex_plus,rex_moderate)
  return(result)
}
q_re <- get_re1(data_filtered,alpha=0.05)


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
  summarise(RV =sum(Ret^2))

har_re_q <- q_re %>%
  select(rex_minus,rex_plus,rex_moderate)%>%
  bind_cols(RV%>%select(RV))




setwd("c:/Users/lenovo/Desktop/HAR")
df <- read.csv("day_data.csv")
# 注意在计算R1,t 的r1,t 的计算和在HAR模型中计算是不一样的，注意这个
#Sti表示的是过去每日收益的回报也就是RETURNs

library(hawkes)

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
# 修改returns中的列名为returns
colnames(returns) <- "returns"


har_pd_re_q <- cbind(har_re_q,returns)

rex_m_lag1 <- lag(har_pd_re_q$rex_minus, 1)
rex_m_lag5 <- rollmean(har_pd_re_q$rex_minus,5,fill=NA,align="right")
rex_m_lag22 <- rollmean(har_pd_re_q$rex_minus,22,fill=NA,align="right")
rex_p_lag1 <- lag(har_pd_re_q$rex_plus, 1)
rex_p_lag5 <- rollmean(har_pd_re_q$rex_plus,5,fill=NA,align="right")
rex_p_lag22 <- rollmean(har_pd_re_q$rex_plus,22,fill=NA,align="right")
rex_moderate_lag1 <- lag(har_pd_re_q$rex_moderate, 1)
rex_moderate_lag5 <- rollmean(har_pd_re_q$rex_moderate,5,fill=NA,align="right")
rex_moderate_lag22 <- rollmean(har_pd_re_q$rex_moderate,22,fill=NA,align="right")

model_data <- data.frame(RV = har_pd_re_q$RV, 
                         rex_m_lag1 = rex_m_lag1, rex_m_lag5 = rex_m_lag5, rex_m_lag22 = rex_m_lag22,
                         rex_p_lag1 = rex_p_lag1, rex_p_lag5 = rex_p_lag5, rex_p_lag22 = rex_p_lag22,
                         rex_moderate_lag1 = rex_moderate_lag1, rex_moderate_lag5 = rex_moderate_lag5, rex_moderate_lag22 = rex_moderate_lag22)
model_data <- na.omit(model_data)

model <- lm(RV ~ ., data = model_data)
model_aic <- AIC(model)
model_bic <- BIC(model)
print(model_aic)
print(model_bic)





test_size <- 800
train_data <- model_data[1:(nrow(model_data) - test_size), ]
test_data <- model_data[(nrow(model_data) - test_size + 1):nrow(model_data), ]


prediction_ree1 <- numeric(nrow(test_data))
prediction_ree5 <- numeric(nrow(test_data))
prediction_ree22 <- numeric(nrow(test_data))
actual_ree1 <- numeric(nrow(test_data))
actual_ree5 <- numeric(nrow(test_data))
actual_ree22 <- numeric(nrow(test_data))


for(i in 1:nrow(test_data)) {
  train_start <- i
  train_end <- i+ nrow(train_data) - 1
  train_windows <- model_data[train_start:train_end, ]
  model <- lm(RV ~ ., data = train_windows)
  prediction_ree1[i] <- predict(model, newdata = model_data[train_end+1, ])
  prediction_ree5[i] <- predict(model, newdata = model_data[train_end+5, ])
  prediction_ree22[i] <- predict(model, newdata = model_data[train_end+22, ])
  actual_ree1[i] <- model_data[train_end+1,1]
  actual_ree5[i] <- model_data[train_end+5,1]
  actual_ree22[i] <- model_data[train_end+22,1]
}
print(sum(prediction_ree1<0))
ree5<- na.omit(prediction_ree5)
ree22 <-na.omit(prediction_ree22)
print(sum(ree5<0))
print(sum(ree22<0))
ree300 <- cbind(actual_ree1,actual_ree5,actual_ree22, prediction_ree1, prediction_ree5, prediction_ree22)
write.csv(ree300,"HARREE800.csv")



test_size <- 600
train_data <- model_data[1:(nrow(model_data) - test_size), ]
test_data <- model_data[(nrow(model_data) - test_size + 1):nrow(model_data), ]


prediction_ree1 <- numeric(nrow(test_data))
prediction_ree5 <- numeric(nrow(test_data))
prediction_ree22 <- numeric(nrow(test_data))
actual_ree1 <- numeric(nrow(test_data))
actual_ree5 <- numeric(nrow(test_data))
actual_ree22 <- numeric(nrow(test_data))


for(i in 1:nrow(test_data)) {
  train_start <- i
  train_end <- i+ nrow(train_data) - 1
  train_windows <- model_data[train_start:train_end, ]
  model <- lm(RV ~ ., data = train_windows)
  prediction_ree1[i] <- predict(model, newdata = model_data[train_end+1, ])
  prediction_ree5[i] <- predict(model, newdata = model_data[train_end+5, ])
  prediction_ree22[i] <- predict(model, newdata = model_data[train_end+22, ])
  actual_ree1[i] <- model_data[train_end+1,1]
  actual_ree5[i] <- model_data[train_end+5,1]
  actual_ree22[i] <- model_data[train_end+22,1]
}
print(sum(prediction_ree1<0))
ree5<- na.omit(prediction_ree5)
ree22 <-na.omit(prediction_ree22)
print(sum(ree5<0))
print(sum(ree22<0))
ree600 <- cbind(actual_ree1,actual_ree5,actual_ree22, prediction_ree1, prediction_ree5, prediction_ree22)
write.csv(ree300,"HARREE600.csv")



#-------------------



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
library(zoo)


loss_function_vectorized <- function(params, data) {
  lanta1 <- params[1]
  beta <- params[2:14]
  
  R1t <- sapply(seq_along(data$returns), function(i) conv_fun1_vectorized(TSPL_kernel_vectorized, data$returns[1:i], lanta1))
  data_df <- data.frame(RV = data$RV, R1t = R1t, REX_m = data$rex_minus, REX_p = data$rex_plus, REX_moderate = data$rex_moderate)
  
  r1t_lag1 <- lag(data_df$R1t, 1)
  r1t_lag5 <- rollmean(data_df$R1t, 5, fill = NA, align = "right")
  r1t_lag22 <- rollmean(data_df$R1t, 22, fill = NA, align = "right")
  rex_m_lag1 <- lag(data_df$REX_m, 1)
  rex_m_lag5 <- rollmean(data_df$REX_m, 5, fill = NA, align = "right")
  rex_m_lag22 <- rollmean(data_df$REX_m, 22, fill = NA, align = "right")
  rex_p_lag1 <- lag(data_df$REX_p, 1)
  rex_p_lag5 <- rollmean(data_df$REX_p, 5, fill = NA, align = "right")
  rex_p_lag22 <- rollmean(data_df$REX_p, 22, fill = NA, align = "right")
  rex_moderate_lag1 <- lag(data_df$REX_moderate, 1)
  rex_moderate_lag5 <- rollmean(data_df$REX_moderate, 5, fill = NA, align = "right")
  rex_moderate_lag22 <- rollmean(data_df$REX_moderate, 22, fill = NA, align = "right")
  data_df <- cbind(data_df,r1t_lag1,r1t_lag5,r1t_lag22)
  data_df <- cbind(data_df,rex_m_lag1,rex_m_lag5,rex_m_lag22)
  data_df <- cbind(data_df,rex_p_lag1,rex_p_lag5,rex_p_lag22)
  data_df <- cbind(data_df,rex_moderate_lag1,rex_moderate_lag5,rex_moderate_lag22)
  
  names(data_df)[6:17] <- c("R1t_1", "R1t_5", "R1t_22", "REX_m_1", "REX_m_5", "REX_m_22", "REX_p_1", "REX_p_5", "REX_p_22", "REX_moderate_1", "REX_moderate_5", "REX_moderate_22")
  col_names <- c("R1t_1", "R1t_5", "R1t_22", "REX_m_1", "REX_m_5", "REX_m_22", "REX_p_1", "REX_p_5", "REX_p_22", "REX_moderate_1", "REX_moderate_5", "REX_moderate_22")
  
  pred_RV <- beta[1] + as.matrix(data_df[, col_names]) %*% beta[-1]
  
  mse <- mean((data_df$RV - pred_RV)^2,na.rm = TRUE)
  rmse <- sqrt(mse)
  return(rmse)
}

initial_params <- runif(14)
opt_result_re <- optim(initial_params, loss_function_vectorized, data = har_pd_re_q, method = "L-BFGS-B",
                       lower = c(1e-7,rep(-3, 13)),
                       upper = c(rep(3, 14)),
                       control = list(maxit = 1000, trace = 1))

# 输出结果
print(opt_result_re$par)
print(opt_result_re$value)
   

#------------------------------------

estimated_params <- opt_result_re$par[1:14]
lanta1 <- estimated_params[1]

R1t <- NULL
for(i in seq(har_pd_re_q$returns)){
  R1t[i] <- conv_fun1_vectorized(TSPL_kernel_vectorized, har_pd_re_q$returns[1:i], lanta1)
}
r1t_lag1 <-lag(R1t,1)
r1t_lag5 <- rollmean(R1t,5,fill=NA,align="right")
r1t_lag22 <- rollmean(R1t,22,fill=NA,align="right")


rex_m_lag1 <-lag(har_re_q$rex_minus,1)
rex_m_lag5 <- rollmean(har_pd_re_q$rex_minus,5,fill=NA,align="right")
rex_m_lag22 <- rollmean(har_pd_re_q$rex_minus,22,fill=NA,align="right")
rex_p_lag1 <-lag(har_pd_re_q$rex_plus,1)
rex_p_lag5 <-rollmean(har_pd_re_q$rex_plus,5,fill=NA,align="right")
rex_p_lag22 <-rollmean(har_pd_re_q$rex_plus,22,fill=NA,align="right")
rex_moderate_lag1 <-lag(har_pd_re_q$rex_moderate,1)
rex_moderate_lag5 <- rollmean(har_pd_re_q$rex_moderate,5,fill=NA,align="right")
rex_moderate_lag22 <- rollmean(har_pd_re_q$rex_moderate,22,fill=NA,align="right")

data_re_pd_q <- data.frame(RV = har_pd_re_q$RV, 
                           rex_m_lag1 = rex_m_lag1, rex_m_lag5 = rex_m_lag5, rex_m_lag22 = rex_m_lag22,
                           rex_p_lag1 = rex_p_lag1, rex_p_lag5 = rex_p_lag5, rex_p_lag22 = rex_p_lag22,
                           rex_moderate_lag1 = rex_moderate_lag1, rex_moderate_lag5 = rex_moderate_lag5, rex_moderate_lag22 = rex_moderate_lag22,
                           r1t_lag1 = r1t_lag1, r1t_lag5 = r1t_lag5, r1t_lag22 = r1t_lag22)
data_repd_q <- na.omit(data_re_pd_q)
split_index <- floor(0.8* nrow(data_repd_q))
train_data <- data_repd_q[1:split_index, ]
test_data <- data_repd_q[(split_index + 1):nrow(data_repd_q), ]
model_re <- lm(RV ~ ., data = train_data)
summary(model_re)
#-----------------------------------------

data_rej <- data.frame(RV = har_re_q$RV, 
                       rex_m_lag1 = rex_m_lag1, rex_m_lag5 = rex_m_lag5, rex_m_lag22 = rex_m_lag22,
                       rex_p_lag1 = rex_p_lag1, rex_p_lag5 = rex_p_lag5, rex_p_lag22 = rex_p_lag22,
                       rex_moderate_lag1 = rex_moderate_lag1, rex_moderate_lag5 = rex_moderate_lag5, rex_moderate_lag22 = rex_moderate_lag22)
data_rej <- na.omit(data_rej)
split_index <- floor(0.8* nrow(data_rej))
train_re <- data_rej[1:split_index, ]
test_re <- data_rej[(split_index + 1):nrow(data_rej), ]
data_rej_model <- lm(RV ~ ., data = train_re)
summary(data_rej_model)


#——---------------- req600

train_df <- data_rej[1:(nrow(data_rej)-1200), ]
test_df <- data_rej[(nrow(data_rej)-1199):nrow(data_rej), ]
last_600 <- numeric(nrow(test_df))

for(i in 1:nrow(test_df)){
  train_start <- i
  train_end <- i + nrow(train_df) - 1
  train_windows <- data_rej[train_start:train_end, ]
  model <- lm(RV ~ ., data = train_windows)
  last_600[i] <- predict(model, newdata = data_rej[train_end + 1, ])
}
out_r2t <- data.frame(RV = test_df$RV, RV_pred = last_600) 
write.csv(out_r2t,"req1200.csv")



#------------------
predictions_re_1 <- numeric(nrow(test_re))
predictions_re_5 <- numeric(nrow(test_re))
predictions_re_22 <- numeric(nrow(test_re))

for(i in 1:nrow(test_re)){
 train_start <- i
 train_end <- i + nrow(train_re) - 1
 train_windows <- data_rej[train_start:train_end, ]
 model <- lm(RV ~ ., data = train_windows)
 predictions_re_1[i] <- predict(model, newdata = data_rej[train_end+1, ])
}
output_data <- data.frame(
  RV_actual = test_re$RV,
  RV_predicted = (predictions_re_1)
)


mse <- mean((test_re$RV - predictions_re_22)^2)
mae <- mean(abs(test_re$RV - predictions_re_22))
log_true <- log(test_re$RV)
log_pred <- log(predictions_re_22)
hmsa <- mean((log_true - log_pred)^2)
hmse <- mean(abs(log_true - log_pred))
qlike <- mean(log_pred + test_re$RV / predictions_re_22)

cat("MSE:",mse,"\n")
cat("MAE:",mae,"\n")
cat("HMSE:",hmse,"\n")
cat("HMSA:",hmsa,"\n")
cat("QLIKE:",qlike,"\n")

output_data <- data.frame(
  RV_actual_22 = test_re$RV,
  RV_predicted_22 = (predictions_re_22)
)


mse_5 <- mean((test_re$RV[1:(nrow(test_re)-21)] - predictions_re_5[1:(nrow(test_re)-21)]^2), na.rm = TRUE)
mae_5 <- mean(abs(test_re$RV[1:(nrow(test_re)-21)] - predictions_re_5[1:(nrow(test_re)-21)]), na.rm = TRUE)
log_true_5 <- log(test_re$RV[1:(nrow(test_re)-21)])
log_pred_5 <- log(predictions_re_5[1:(nrow(test_re)-21)])
hmse_5 <- mean(abs(log_true_5 - log_pred_5), na.rm = TRUE)
hmsa_5 <- mean((log_true_5 - log_pred_5)^2, na.rm = TRUE)
qlike_5 <- mean(log_pred_5 + test_re$RV[1:(nrow(test_re)-21)] / predictions_re_5[1:(nrow(test_re)-21)], na.rm = TRUE)

mse_22 <- mean((test_re$RV[1:(nrow(test_re)-21)] - predictions_re_22[1:(nrow(test_re)-21)]^2), na.rm = TRUE)
mae_22 <- mean(abs(test_re$RV[1:(nrow(test_re)-21)] - predictions_re_22[1:(nrow(test_re)-21)]), na.rm = TRUE)
log_true_22 <- log(test_re$RV[1:(nrow(test_re)-21)])
log_pred_22 <- log(predictions_re_22[1:(nrow(test_re)-21)])
hmse_22 <- mean(abs(log_true_22 - log_pred_22), na.rm = TRUE)
hmsa_22 <- mean((log_true_22 - log_pred_22)^2, na.rm = TRUE)
qlike_22 <- mean(log_pred_22 + test_re$RV[1:(nrow(test_re)-21)] / predictions_re_22[1:(nrow(test_re)-21)], na.rm = TRUE)

cat("5步预测损失值:\n")
cat("MSE:",mse_5,"\n")
cat("MAE:",mae_5,"\n")
cat("HMSE:",hmse_5,"\n")
cat("HMSA:",hmsa_5,"\n")
cat("QLIKE:",qlike_5,"\n\n")

cat("22步预测损失值:\n")
cat("MSE:",mse_22,"\n")
cat("MAE:",mae_22,"\n")
cat("HMSE:",hmse_22,"\n")
cat("HMSA:",hmsa_22,"\n")
cat("QLIKE:",qlike_22,"\n")

output_data <- data.frame(
  RV_actual_5 = test_re$RV,
  RV_predicted_5 = (predictions_re_5), 
  RV_actual_22 = test_re$RV,
  RV_predicted_22 = (predictions_re_22)
)

#------------------------------- har_re_pd_q

estimated_params <- opt_result_re$par[1:14]
lanta1 <- estimated_params[1]

R1t <- NULL
for(i in seq(har_pd_re_q$returns)){
  R1t[i] <- conv_fun1_vectorized(TSPL_kernel_vectorized, har_pd_re_q$returns[1:i], 0.0000001)
}
r1t_lag1 <-lag(R1t,1)
r1t_lag5 <- rollmean(R1t,5,fill=NA,align="right")
r1t_lag22 <- rollmean(R1t,22,fill=NA,align="right")
data_re_pd_q <- data.frame(RV = har_pd_re_q$RV, 
                         rex_m_lag1 = rex_m_lag1, rex_m_lag5 = rex_m_lag5, rex_m_lag22 = rex_m_lag22,
                         rex_p_lag1 = rex_p_lag1, rex_p_lag5 = rex_p_lag5, rex_p_lag22 = rex_p_lag22,
                         rex_moderate_lag1 = rex_moderate_lag1, rex_moderate_lag5 = rex_moderate_lag5, rex_moderate_lag22 = rex_moderate_lag22,
                         r1t_lag1 = r1t_lag1, r1t_lag5 = r1t_lag5, r1t_lag22 = r1t_lag22)
data_repd_q <- na.omit(data_re_pd_q)
split_index <- floor(0.8* nrow(data_repd_q))
train_data <- data_repd_q[1:split_index, ]
test_data <- data_repd_q[(split_index + 1):nrow(data_repd_q), ]
model_re <- lm(RV ~ ., data = train_data)
summary(model_re)

predictions_re_q <- numeric(nrow(test_data))

for(i in 1:nrow(test_data)){
  train_start <- i
  train_end <- i + nrow(train_data) - 1
  train_windows <- data_repd_q[train_start:train_end, ]
  predictions_re_q[i] <- predict(model_re, newdata = data_repd_q[train_end+1, ])
}
output_data <- data.frame(
  RV_actual = test_data$RV,
  RV_predicted = (predictions_re_q)
)
write.csv(output_data,"newreqpd.csv")


train_df <- data_repd_q[1:(nrow(data_repd_q)-600), ]
test_df <- data_repd_q[(nrow(data_repd_q)-599):nrow(data_repd_q), ]
last_600 <- numeric(nrow(test_df))

for(i in 1:nrow(test_df)){
  train_start <- i
  train_end <- i + nrow(train_df) - 1
  train_windows <- data_repd_q[train_start:train_end, ]
  model <- lm(RV ~ ., data = train_windows)
  last_600[i] <- predict(model, newdata = data_repd_q[train_end + 1, ])
}
out_r2t <- data.frame(RV = test_df$RV, RV_pred = last_600) 
write.csv(out_r2t,"reqpd600.csv")



mse <- mean((test_data$RV - predictions_re_q)^2)
mae <- mean(abs(test_data$RV - predictions_re_q))

log_true <- log(test_data$RV)
log_pred <- log(predictions_re_q)
hmsa <- mean((log_true - log_pred)^2)
hmse <- mean(abs(log_true - log_pred))
qlike <- mean(log_pred + test_data$RV / predictions_re_q)

cat("MSE:",mse,"\n")
cat("MAE:",mae,"\n")
cat("HMSE:",hmse,"\n")
cat("HMSA:",hmsa,"\n")
cat("QLIKE:",qlike,"\n")


predictions_reqpd_1 <- numeric(nrow(test_data))
predictions_reqpd_5 <- numeric(nrow(test_data))
predictions_reqpd_22 <- numeric(nrow(test_data))


  
for(i in 1:nrow(test_data)){
  train_start <- i
  train_end <- nrow(train_data) + i - 1
  train_windows <- data_repd_q[train_start:train_end, ]
  model <- lm(RV ~ . ,data = train_windows)
  predictions <- predict(model, newdata = data_repd_q[train_end+1, ])
  predictions_reqpd_1[i] <- predictions
    predictions_reqpd_5[i] <- predict(model, newdata = data_repd_q[train_end + 5, ])
    predictions_reqpd_22[i] <- predict(model, newdata = data_repd_q[train_end + 22, ])
}

output_data <- data.frame(
  RV_actual_5 = test_data$RV,
  RV_predicted_5 = (predictions_reqpd_5), 
  RV_actual_22 = test_data$RV,
  RV_predicted_22 = (predictions_reqpd_22)
)


mse_5 <- mean((test_data$RV[1:(nrow(test_data)-4)]- predictions_reqpd_5[1:(nrow(test_data)-4)]^2),na.rm = TRUE)
mae_5 <- mean(abs(test_data$RV[1:(nrow(test_data)-4)]- predictions_reqpd_5[1:(nrow(test_data)-4)]),na.rm = TRUE)
log_true_5 <- log(test_data$RV[1:(nrow(test_data)-4)])
log_pred_5 <- log(predictions_reqpd_5[1:(nrow(test_data)-4)])
hmse_5 <- mean(abs(log_true_5 - log_pred_5),na.rm = TRUE)
hmsa_5 <- mean((log_true_5 - log_pred_5)^2,narm = TRUE)
qlike_5 <- mean(log_pred_5 + test_data$RV[1:(nrow(test_data)-4)] / predictions_reqpd_5[1:(nrow(test_data)-4)],na.rm = TRUE)

mse_22 <- mean((test_data$RV[1:(nrow(test_data)-21)]- predictions_reqpd_22[1:(nrow(test_data)-21)]^2),na.rm = TRUE)
mae_22 <- mean(abs(test_data$RV[1:(nrow(test_data)-21)]- predictions_reqpd_22[1:(nrow(test_data)-21)]),na.rm = TRUE)
log_true_22 <- log(test_data$RV[1:(nrow(test_data)-21)])
log_pred_22 <- log(predictions_reqpd_22[1:(nrow(test_data)-21)])
hmse_22 <- mean(abs(log_true_22 - log_pred_22),na.rm = TRUE)
hmsa_22 <- mean((log_true_22 - log_pred_22)^2,narm = TRUE)
qlike_22 <- mean(log_pred_22 + test_data$RV[1:(nrow(test_data)-21)] / predictions_reqpd_22[1:(nrow(test_data)-21)],na.rm = TRUE)


cat("5步预测损失值:\n")
cat("MSE:",mse_5,"\n")
cat("MAE:",mae_5,"\n")
cat("HMSE:",hmse_5,"\n")
cat("HMSA:",hmsa_5,"\n")
cat("QLIKE:",qlike_5,"\n\n")

cat("22步预测损失值:\n")
cat("MSE:",mse_22,"\n")
cat("MAE:",mae_22,"\n")
cat("HMSE:",hmse_22,"\n")
cat("HMSA:",hmsa_22,"\n")
cat("QLIKE:",qlike_22,"\n")



