
setwd("c:/Users/lenovo/Desktop/HAR")
df <- read.csv("data_idx.csv")
library(dplyr)
library(tidyverse)
library(xts)
# 按组进行分类
group_summary <- df %>%
 group_by(code) %>%
 summarise(NumObservations = n())

# 提取 code 为 "000001.XSHG" 的部分
data_filtered <- df %>%
 filter(code == "000001.XSHG")
library(dplyr)
library(lubridate)
library(dplyr)
library(lubridate)



get_RV_BV <- function(data, alpha = 0.05, times = TRUE){
  if(times){
    idx <- 100
  } else {
    idx <- 1
  }
  
  result <- data %>% 
    mutate(
      # 确保 time 列为字符类型
      time = as.character(time),
      # 提取前 10 个字符
      date_str = substr(time, 1, 10),
      # 使用 as.Date() 转换日期
      day = as.Date(date_str)
    ) %>% 
    # 检查 day 列中是否有 NA 值
    filter(!is.na(day)) %>% 
    group_by(day) %>% 
    mutate(
      Ret = (log(close) - log(dplyr::lag(close))) * idx
    ) %>% 
    na.omit() %>% 
    summarise(
      RV = sum(Ret^2),
      BV = (pi/2) * sum(na.omit( lag(abs(Ret)) * lead(abs(Ret)) )),
      TQ = n() * (2^(2/3) * gamma(7/6) * gamma(1/2)^(-1))^(-3) * (n()/(n() - 4)) * sum(abs(Ret[5:n()])^(4/3) * abs(Ret[3:(n() - 2)])^(4/3) * abs(Ret[1:(n() - 4)])^(4/3)),
      Z_test = ((RV - BV)/RV) / ( ((pi/2)^2 + pi - 5) * (n())^(-1) * max(1, TQ/(BV^2)) )^(1/2),
      JV = (RV - BV) * (Z_test > qnorm(1 - alpha)),
      C_t = ifelse(Z_test <= qnorm(1 - alpha), RV, BV)
    ) %>% 
    select(day, RV, BV, JV, C_t)
  
  return(result)
}
har_cj <- get_RV_BV(data_filtered,alpha = 0.05,times=FALSE)
print(har_cj)

#------------------------------

setwd("c:/Users/lenovo/Desktop/HAR")
df <- read.csv("day_data.csv")
# 注意在计算R1,t 的r1,t 的计算和在HAR模型中计算是不一样的，注意这个
#Sti表示的是过去每日收益的回报也就是RETURNs
library(tidyverse)
library(hawkes)
library(highfrequency)
library(dplyr)
library(xts)

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


# 进行分类并提取第一类数据
data_filtered <- df %>%
 dplyr::select(time, code, close) %>%
 set_names(c("DT", "id", "PRICE")) %>%
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
returns <- ret(data_filtered$PRICE)
returns <- as.data.frame(returns)




# 修改returns中的列名为returns
colnames(returns) <- "returns"
#--------------------------




#-----------------------计算RV
setwd("c:/Users/lenovo/Desktop/HAR")
df <- read.csv("data_idx.csv")

# 按组进行分类
group_summary <- df %>%
  group_by(code) %>%
  summarise(NumObservations = n())

# 提取 code 为 "000001.XSHG" 的部分
data_filtered <- df %>%
  filter(code == "000001.XSHG")


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

data_get_cj <- data.frame(RV=RV$RV,BV=har_cj$BV,JV=har_cj$JV,CT=har_cj$C_t,returns=returns$returns)  

#---------------------------计算har-cj模型

library(HARModel)
# 定义训练集和测试集
total_rows <- nrow(data_get_cj)
split_point <- total_rows - 300
train_data <- data_get_cj[1:split_point, ]
test_data <-data_get_cj[(split_point + 1):total_rows, ]
harcj <- HAREstimate(data_get_cj$RV,data_get_cj$BV,periods=c(1,5,22),periodsJ=c(1,5,22),type="HARJ")
summary(harcj)

HARFore1 <- HARForecast(train_data$RV,train_data$BV,periods=c(1,5,22),periodsJ=c(1,5,22),
                        nRoll=300,nAhead=1,type="HARJ")
cj1 <- getForc(HARFore1)
write.csv(cj1,"harcj1300.csv")

HARFore5 <- HARForecast(train_data$RV,train_data$BV,periods=c(1,5,22),periodsJ=c(1,5,22),
                        nRoll=297,nAhead=5,type="HARJ")
cj5 <- getForc(HARFore5)
write.csv(cj5,"harcj5300.csv")
HARFore22 <- HARForecast(train_data$RV,train_data$BV,periods=c(1,5,22),periodsJ=c(1,5,22),
                        nRoll=278,nAhead=22,type="HARJ")
cj22 <- getForc(HARFore22)
write.csv(cj22,"harcj22300.csv")








cj_lag1 <- lag(data_get_cj$JV,1)
cj_lag5 <- rollmean(data_get_cj$JV,5,align="right",fill=NA)
cj_lag22 <- rollmean(data_get_cj$JV,22,align="right",fill=NA)
ct_lag1 <- lag(data_get_cj$CT,1)
ct_lag5 <- rollmean(data_get_cj$CT,5,align="right",fill=NA)
ct_lag22 <- rollmean(data_get_cj$CT,22,align="right",fill=NA)
model_data <- data.frame(RV=data_get_cj$RV,cj_lag1,cj_lag5,cj_lag22,ct_lag1,ct_lag5,ct_lag22)
model_data <- na.omit(model_data)
model <-lm(RV~.,data=model_data)
model_aic<- AIC(model)
model_bic <- BIC(model)
print(model_aic)
print(model_bic)



test_size <- 1000
train_data <- model_data[1:(nrow(model_data) - test_size), ]
test_data <- model_data[(nrow(model_data) - test_size + 1):nrow(model_data), ]


predictioncj1 <- numeric(nrow(test_data))
predictioncj5 <- numeric(nrow(test_data))
predictioncj22 <- numeric(nrow(test_data))
for(i in 1:nrow(test_data)){
  train_start <- i
  train_end <- i + nrow(train_data) - 1
  train_windows <- model_data[train_start:train_end,]
  model <- lm(RV ~ .,data=train_windows)
  predictioncj1[i] <- predict(model,newdata=model_data[train_end+1,])
  predictioncj5[i] <- predict(model,newdata=model_data[train_end+5,])
  predictioncj22[i] <- predict(model,newdata=model_data[train_end+22,])
}
cj300 <- cbind(predictioncj1,predictioncj5,predictioncj22)
write.csv(cj300,"HARCJ800.csv")



#--------------------------------------
# 已经划分好数据集，进去计算
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





estimated_params <- opt_result_cj$par[1:11]
lanta1 <- estimated_params[1]

#9.620838e-01
R2t <- NULL
for(i in seq(har_cj_r$returns)){
 R2t[i] <- conv_fun2_vectorized(TSPL_kernel_vectorized, har_cj_r$returns[1:i], 
                                9.620838e-01)
}

r2t <- lag(R2t, 1)
cj <- lag(data_get_cj$JV, 1)
cv <- lag(data_get_cj$CT, 1)

r2t5 <- rollmean(R2t, 5, align = "right", fill = NA)
cj5 <- rollmean(data_get_cj$JV, 5, align = "right", fill = NA)
ct5 <- rollmean(data_get_cj$CT, 5, align = "right", fill = NA)

r2t22 <- rollmean(R2t, 22, align = "right", fill = NA)
cj22 <- rollmean(data_get_cj$JV, 22, align = "right", fill = NA)
ct22 <- rollmean(data_get_cj$CT, 22, align = "right", fill = NA)


data_har_cj <- data.frame(RV=data_get_cj$RV,R2t=r2t, R2t5 = r2t5, R2t22 = r2t22,
                          CJ_lag1 = cj, CJ_lag5 = cj5, CJ_lag22 = cj22, 
                          CT_lag1 = cv, CT_lag5 = ct5, CT_lag22 = ct22)
data_har_cj <- na.omit(data_har_cj)
split_index <- floor(0.8*nrow(data_har_cj))
train_data_har_cj <- data_har_cj[(1:split_index),]
test_data_har_cj <- data_har_cj[(split_index+1):nrow(data_har_cj),]
har_cj <- lm(RV ~ .,data=train_data_har_cj)
summary(har_cj)


ucj <- data.frame(RV=data_get_cj$RV,
                          CJ_lag1 = cj, CJ_lag5 = cj5, CJ_lag22 = cj22, 
                          CT_lag1 = ct, CT_lag5 = ct5, CT_lag22 = ct22)
ucj <- na.omit(ucj)
split_index <- floor(0.8*nrow(ucj))
train_data_ucj <- ucj[(1:split_index),]
test_data_ucj <- ucj[(split_index+1):nrow(ucj),]
har_ucj <- lm(RV ~ .,data=train_data_ucj)
summary(har_ucj)


preidction_cj <- numeric(nrow(test_data_har_cj))
for(i in 1:nrow(test_data_har_cj)){
  train_start <- i
  train_end <- i + nrow(train_data_har_cj) - 1
  train_windows <- data_har_cj[train_start:train_end,]
  model <- lm(RV ~ ., data = train_windows)
  preidction_cj[i] <- predict(model, newdata = data_har_cj[train_end + 1,])
}
prediction_cj1 <- numeric(nrow(test_data_har_cj))
prediction_cj5 <- numeric(nrow(test_data_har_cj))
prediction_cj22 <- numeric(nrow(test_data_har_cj))
for (i in 1:nrow(test_data_har_cj)) {
  train_start <- i
  train_end <- i + nrow(train_data_har_cj) - 1
  train_windows <- data_har_cj[train_start:train_end, ]
  model <- lm(RV ~ ., data = train_windows)
  
  prediction_cj1[i] <- predict(model, newdata = data_har_cj[train_end + 1, ])
  prediction_cj5[i] <- predict(model, newdata = data_har_cj[train_end + 5, ])
  prediction_cj22[i] <- predict(model, newdata = data_har_cj[train_end + 22, ])
}
out <- data.frame(test = test_data_har_cj$RV, prediction_cj1, prediction_cj5, prediction_cj22)
write.csv(out,"1522.csv")


predictions_cj <- numeric(nrow(test_data_r12))
for(i in 1:nrow(test_data_r12)){
  train_start <- i
  train_end <- i + nrow(train_data_r12) -1
  train_windows <- data_r12[train_start:train_end,]
  model <- lm(RV~., data = train_windows)
  predictions_cj[i] <- predict(model, newdata = data_r12[train_end+22,])
}

prdeictions <- numeric(nrow(Y_test_data_r12))
for(i in 1:nrow(Y_test_data)){
  train_start <- i 
  train_end <- i + nrow(X_trian_data)-1
  train_windows <- 
}



output <- data.frame(test = test_data_r12$RV, predictions_cj)


cv_lag1 <- lag(har_cj$C_t,1)
cv_lag5 <- rollmean(har_cj$C_t,6,align="right",fill=NA)
cv_lag22 <- rollmean(har_cj$C_t,23,align="right",fill=NA)
cj_lag1 <- lag(har_cj$JV,1)
cj_lag5 <- rollmean(har_cj$JV,6,align="right",fill=NA)
cj_lag22 <- rollmean(har_cj$JV,23,align="right",fill=NA)

data12 <- data.frame(RV= data_har_cj$RV,cv_lag1,cv_lag5,cv_lag22,cj_lag1,cj_lag5,cj_lag22)
data12 <- na.omit(data12)
split_index <- floor(0.9*nrow(data12))
train_data12 <- data12[(1:split_index),]
test_data12 <- data12[(split_index+1):nrow(data12),]
har12 <- lm(RV ~ .,data=train_data12)
summary(har12)

predictions_cj1 <- numeric(nrow(test_data12))
predictions_cj5 <- numeric(nrow(test_data12))
predictions_cj22 <- numeric(nrow(test_data12))

for(i in 1:nrow(test_data12)){
  train_start <- i
  train_end <- i + nrow(train_data12) - 1
  train_windows <- data12[train_start:train_end,]
  model <- lm(RV ~ ., data = train_windows)
  
  predictions_cj1[i] <- predict(model, newdata = data12[train_end + 1,])
    predictions_cj5[i] <- predict(model, newdata = data12[train_end + 5,])
    predictions_cj22[i] <- predict(model, newdata = data12[train_end + 22,])
}
write.csv(predictions_cj22,"predictions_cj22.csv")
mse_5 <- mean((test_data12$RV[1:(nrow(test_data12)-4)] - predictions_cj5[1:(nrow(test_data12)-4)])^2, na.rm = TRUE)
mae_5 <- mean(abs(test_data12$RV[1:(nrow(test_data12)-4)] - predictions_cj5[1:(nrow(test_data12)-4)]), na.rm = TRUE)
log_true_5 <- log(test_data12$RV[1:(nrow(test_data12)-4)])
log_pred_5 <- log(predictions_cj5[1:(nrow(test_data12)-4)])
hmse_5 <- mean(abs(log_true_5 - log_pred_5), na.rm = TRUE)
hmsa_5 <- mean((log_true_5 - log_pred_5)^2, na.rm = TRUE)
qlike_5 <- mean(log_pred_5 + test_data12$RV[1:(nrow(test_data12)-4)] / predictions_cj5[1:(nrow(test_data12)-4)], na.rm = TRUE)

# 计算未来22步的损失值
mse_22 <- mean((test_data12$RV[1:(nrow(test_data12)-21)] - predictions_cj22[1:(nrow(test_data12)-21)])^2, na.rm = TRUE)
mae_22 <- mean(abs(test_data12$RV[1:(nrow(test_data12)-21)] - predictions_cj22[1:(nrow(test_data12)-21)]), na.rm = TRUE)
log_true_22 <- log(test_data12$RV[1:(nrow(test_data12)-21)])
log_pred_22 <- log(predictions_cj22[1:(nrow(test_data12)-21)])
hmse_22 <- mean(abs(log_true_22 - log_pred_22), na.rm = TRUE)
hmsa_22 <- mean((log_true_22 - log_pred_22)^2, na.rm = TRUE)
qlike_22 <- mean(log_pred_22 + test_data12$RV[1:(nrow(test_data12)-21)] / predictions_cj22[1:(nrow(test_data12)-21)], na.rm = TRUE)

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

output <- data.frame(test5 =test_data12, predictions_cj5,test22=test_data12, predictions_cj22)



split_index <- floor(0.8*nrow(RV))
train <- RV[(1:split_index),]
test <- RV[(split_index+1):nrow(RV),]



print(sum(predictions_cj<0))
mse <- mean((test$RV - predictions_cj)^2)
mae <- mean(abs(test$RV - predictions_cj))
log_true <- log(test$RV)
log_pred <- log(predictions_cj)
hmse <- mean((log_true - log_pred)^2)
hmsa <- mean(abs(log_true - log_pred))
qlike <- mean(log_pred+test$RV/predictions_cj)


cat("MSE: ", mse, "\n")
cat("MAE: ", mae, "\n")
cat("HMSE: ", hmse, "\n")
cat("HMSA: ", hmsa, "\n")
cat("QLIKE:",qlike,"\n")
print(opt_result_cj$par)
print(opt_result_cj$value)





predictions_cj_1 <- numeric(nrow(test_data_r12))
predictions_cj_5 <- numeric(nrow(test_data_r12))
predictions_cj_22 <- numeric(nrow(test_data_r12))

for(i in 1:nrow(test_data_r12)){
  train_start <- i
  train_end <- i + nrow(train_data_r12) - 1
  train_windows <- data_r12[train_start:train_end, ]
  model <- lm(RV ~ ., data = train_windows)
  
  predictions_cj_1[i] <- predict(model, newdata = data_r12[train_end + 1, ])
    predictions_cj_5[i] <- predict(model, newdata = data_r12[train_end + 5, ])
    predictions_cj_22[i] <- predict(model, newdata = data_r12[train_end + 22, ])
}
mse_5 <- mean((test_data_r12$RV[1:(nrow(test_data_r12)-4)] - predictions_cj_5[1:(nrow(test_data_r12)-4)])^2, na.rm = TRUE)
mae_5 <- mean(abs(test_data_r12$RV[1:(nrow(test_data_r12)-4)] - predictions_cj_5[1:(nrow(test_data_r12)-4)]), na.rm = TRUE)
log_true_5 <- log(test_data_r12$RV[1:(nrow(test_data_r12)-4)])
log_pred_5 <- log(predictions_cj_5[1:(nrow(test_data_r12)-4)])
hmse_5 <- mean(abs(log_true_5 - log_pred_5), na.rm = TRUE)
hmsa_5 <- mean((log_true_5 - log_pred_5)^2, na.rm = TRUE)
qlike_5 <- mean(log_pred_5 + test_data_r12$RV[1:(nrow(test_data_r12)-4)] / predictions_cj_5[1:(nrow(test_data_r12)-4)], na.rm = TRUE)

# 计算未来22步的损失值
mse_22 <- mean((test_data_r12$RV[1:(nrow(test_data_r12)-21)] - predictions_cj_22[1:(nrow(test_data_r12)-21)])^2, na.rm = TRUE)
mae_22 <- mean(abs(test_data_r12$RV[1:(nrow(test_data_r12)-21)] - predictions_cj_22[1:(nrow(test_data_r12)-21)]), na.rm = TRUE)
log_true_22 <- log(test_data_r12$RV[1:(nrow(test_data_r12)-21)])
log_pred_22 <- log(predictions_cj_22[1:(nrow(test_data_r12)-21)])
hmse_22 <- mean(abs(log_true_22 - log_pred_22), na.rm = TRUE)
hmsa_22 <- mean((log_true_22 - log_pred_22)^2, na.rm = TRUE)
qlike_22 <- mean(log_pred_22 + test_data_r12$RV[1:(nrow(test_data_r12)-21)] / predictions_cj_22[1:(nrow(test_data_r12)-21)], na.rm = TRUE)


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


train_df <- data_r12[1:(nrow(data_r12)-1200), ]
test_df <- data_r12[(nrow(data_r12)-1199):nrow(data_r12), ]
last_600 <- numeric(nrow(test_df))

for(i in 1:nrow(test_df)){
  train_start <- i
  train_end <- i + nrow(train_df) - 1
  train_windows <- data_r12[train_start:train_end, ]
  model <- lm(RV ~ ., data = train_windows)
  last_600[i] <- predict(model, newdata = data_r12[train_end + 1, ])
}
out_r2t <- data.frame(RV = test_df$RV, RV_pred = last_600) 
write.csv(out_r2t,"cjpd1200.csv")
output <- data.frame(test1 =test_data_r12$RV,predictions_cj_1,test5 =test_data_r12$RV, predictions_cj_5,test22=test_data_r12$RV, predictions_cj_22)
write.csv(output,"newharcj.csv")
#提取har_cj中的RV列和JV列,再与returns合并


train_df <- data_get_cj[1:(nrow(data_get_cj)-1200), ]
test_df <- data_get_cj[(nrow(data_get_cj)-1199):nrow(data_get_cj), ]
last_600 <- numeric(nrow(test_df))

HARFore1 <- HARForecast(train_df$RV,train_df$BV,periods=c(1,5,22),periodsJ=c(1,5,22),
                        nRoll=1200,nAhead=1,type="HARJ")
c1 <- getForc(HARFore1)
write.csv(c1,"HARForecj1200.csv")


library(HARModel)
split_index <- floor(0.8*nrow(har_cj))
train <- har_cj[(1:split_index),]
test <- har_cj[(split_index+1):nrow(har_cj),]

HARFore2 <- HARForecast(train$RV,train$BV,periods=c(1,5,22),periodsJ=c(1,5,22),
                        nRoll=916,nAhead=1,type="HARJ")
c2 <- getForc(HARFore2)
write.csv(c2,"HARForecjyuanbenyibu.csv")

#-------------------------------- 以上代码是经过了变换以后的回归，现在做一个基础回归预测
# 
# library(dplyr)
# library(zoo) 
# har_cj_b <- har_cj %>%
#  dplyr::select(RV, JV, C_t) %>%
#  dplyr::rename(RV = RV, CJ = JV, CV = C_t)
# 
# split_index <- floor(nrow(har_cj_b) * 0.8)
# train_data_b <- har_cj_b[1:split_index, ]
# test_data_b <- har_cj_b[(split_index + 1):nrow(har_cj_b), ]
# len_train <- nrow(train_data_b)
# len_test <- nrow(test_data_b)
# min_length <- min(len_train, len_test)
# ts_data <- train_data_b[(len_train - min_length + 1):len_train, ]
# #-----------判断ts_data 和 train_data_b$RV中多少数值相等
# 
# 
# # 计算滞后和滚动平均
# train_data_b <- train_data_b %>%
#  mutate(
#   CJ_lag1 = lag(CJ, 1),
#   CJ_lag5 = rollmean(CJ, 5, align = "right", fill = NA),
#   CJ_lag22 = rollmean(CJ, 22, align = "right", fill = NA),
#   CV_lag1 = lag(CV, 1),
#   CV_lag5 = rollmean(CV, 5, align = "right", fill = NA),
#   CV_lag22 = rollmean(CV, 22, align = "right", fill = NA)
#  )
# 
# # 移除包含NA的行
# train_data_b <- na.omit(train_data_b)
# 
# har_cj_model <- lm(RV ~ CJ_lag1 + CJ_lag5 + CJ_lag22 + CV_lag1 + CV_lag5 + CV_lag22, data = train_data_b)
# 
# summary(har_cj_model)
# 
# #--------------------滚动窗口预测
# coe <- summary(har_cj_model)$coef
# cof <- coe[,1]
# windows <- c(1, 5, 22)
# 
# rolling_predictions <- vector("numeric", length = nrow(ts_data))
# for (i in 1:nrow(ts_data)){
#  if (i == 1) {
#   updated_data_rt <- ts_data
#  } else {
#   updated_data_rt <- rbind(ts_data[i:nrow(ts_data), ], test_data_b[1:(i-1), ])
#  }
#  CJ_lag1 <- updated_data_rt$CJ[nrow(updated_data_rt)]
#  CV_lag1 <- updated_data_rt$CV[nrow(updated_data_rt)]
#  
#  
#  CJ_lag5 <- mean(updated_data_rt$CJ[(nrow(updated_data_rt)-windows[2]+1):nrow(updated_data_rt)])
#  CV_lag5 <- mean(updated_data_rt$CV[(nrow(updated_data_rt)-windows[2]+1):nrow(updated_data_rt)])
#  
#  
#  CJ_lag22 <- mean(updated_data_rt$CJ[(nrow(updated_data_rt)-windows[3]+1):nrow(updated_data_rt)])
#  CV_lag22 <- mean(updated_data_rt$CV[(nrow(updated_data_rt)-windows[3]+1):nrow(updated_data_rt)])
#  
#  estimated_params <- cof[1:6]
#  predicted_RVt <- estimated_params[1] +
#   estimated_params[1] * (CJ_lag1) +
#   estimated_params[2] * (CJ_lag5) +
#   estimated_params[3] * (CJ_lag22) +
#   estimated_params[4] * (CV_lag1) +
#   estimated_params[5] * (CV_lag5) +
#   estimated_params[6] * (CV_lag22)
#  rolling_predictions[i] <-   predicted_RVt
# }
# print(rolling_predictions)
# 
# 
# mse <- mean((test_data_b$RV - rolling_predictions)^2)
# mae <- mean(abs(test_data_b$RV - rolling_predictions))
# 
# log_true <- log(test_data_b$RV)
# log_pred <- log(rolling_predictions)
# hmse <- mean((log_true - log_pred)^2)
# hmsa <- mean(abs(log_true - log_pred))
# qlike <- mean(log_pred+test_data_b$RV/rolling_predictions)
# cat("MSE: ", mse, "\n")
# cat("MAE: ", mae, "\n")
# cat("HMSE: ", hmse, "\n")
# cat("HMSA: ", hmsa, "\n")
# cat("QLIKE:",qlike,"\n")
# 
