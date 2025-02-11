

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



#------------------------------- har_re_pd_q



