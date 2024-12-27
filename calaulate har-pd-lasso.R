

df <- read.csv("xxx.csv")
library(dplyr)
library(tidyverse)
library(xts)
group_summary <- df %>%
  group_by(code) %>%
  summarise(NumObservations = n())

# time series data
data_filtered <- df %>%
  filter(code == "xxx.XSHG")
library(dplyr)
library(lubridate)
library(dplyr)
library(lubridate)


# calculate har-cj model
get_RV_BV <- function(data, alpha = 0.05, times = TRUE){
  if(times){
    idx <- 100
  } else {
    idx <- 1
  }
  
  result <- data %>% 
    mutate(
      time = as.character(time),
      date_str = substr(time, 1, 10),
      day = as.Date(date_str)
    ) %>% 
    
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

df <- read.csv("day_data.csv")
# attention: the data should be sorted by time
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


group_summary <- data_ret %>% 
  group_by(id) %>%
  summarise(NumObservations = n())


# filter the data
data_filtered <- df %>%
  dplyr::select(time, code, close) %>%
  set_names(c("DT", "id", "PRICE")) %>%
  na.omit() %>%
  filter(id == "xxx.XSHG")


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




# rename the columns
colnames(returns) <- "returns"
#-------------------------



#-----------------------calculate the RV and BV

df <- read.csv("xxx.csv")

# filter the data
group_summary <- df %>%
  group_by(code) %>%
  summarise(NumObservations = n())

# 
data_filtered <- df %>%
  filter(code == "xxx.XSHG")


data_ret <- df %>%
  select(time, code, close) %>%
  set_names(c("DT", "id", "PRICE")) %>%
  na.omit() %>%
  group_split(id) %>%
  lapply(., function (x) data.frame(DT = x$DT, id=x$id,Ret = makeReturns(x$PRICE)))%>%
  bind_rows()

# filter the data
group_summary <- data_ret %>%
  group_by(id) %>%
  summarise(NumObservations = n())

# Create a data frame that can be used for HAR and extract the first set
data_filtered <- data_ret %>%
  filter(id == "xxx.XSHG")
data_filtered <- data_filtered[,-4]


data_filtered$DT <- as.Date(data_filtered$DT)
RV <- data_filtered %>%
  group_by(DT) %>%
  summarise(RV =sum(Ret^2))

data_get_cj <- data.frame(RV=RV$RV,X,returns=returns$returns)  



#--------------------------------------

TSPL_kernel_vectorized <- function(diff_times, lanta) {
  lanta * exp(-lanta * diff_times)
}

conv_fun1_vectorized <- function(kernel, x, lanta) {
  diff_times <- seq_along(x) - 1
  weights <- rev(kernel(diff_times, lanta))
  normalized_weights <- weights / sum(weights)  
  sum(normalized_weights * x)
}
conv_fun2_vectorized <- function(kernel, x, lanta) {
  diff_times <- seq_along(x) - 1
  weights <- rev(kernel(diff_times, lanta))
  normalized_weights <- weights / sum(weights)  
  sum(normalized_weights * x^2)
}


library(glmnet)
library(zoo)  # Added for rollmean function


# Step 1: Loss function for optimizing lanta1, lanta2, lanta3
loss_function_lanta <- function(params, data) {
  # Extract lanta1, lanta2, lanta3
  lanta1 <- params[1]
  lanta2 <- params[2]
  lanta3 <- params[3]
  
  # Calculate measures
  x1<- sapply(seq_along(data$returns), function(i) 
    conv_fun2_vectorized(TSPL_kernel_vectorized, data$returns[1:i], lanta1))
  
  x2 <- sapply(seq_along(data$returns), function(i) 
    conv_fun1_vectorized(TSPL_kernel_vectorized, data$JV[1:i], lanta2))
  
  x3 <- sapply(seq_along(data$returns), function(i) 
    conv_fun1_vectorized(TSPL_kernel_vectorized, data$CT[1:i], lanta3))
  
  # Create model matrix
  model_data <- create_model_matrix(data, r2t, cjt, cvt)
  X <- as.matrix(model_data[, -1])
  y <- model_data$RV
  
  # Perform cross-validation for LASSO
  tryCatch({
    cv_model <- cv.glmnet(X, y, alpha = 1, nfolds = 10)
    lambda_min <- cv_model$lambda.min
    lasso_model <- glmnet(X, y, alpha = 1, lambda = lambda_min)
    
    # Predict and compute adjusted R-squared
    y_pred <- predict(lasso_model, X)
    residuals <- y - y_pred
    tss <- sum((y - mean(y))^2)
    rss <- sum(residuals^2)
    n <- length(y)
    p <- sum(coef(lasso_model) != 0) - 1  # Non-zero coefficients
    r_squared_adj <- 1 - (rss / (n - p - 1)) / (tss / (n - 1))
    
    return(-r_squared_adj)  # Return negative for minimization
  }, error = function(e) {
    warning("Error in LASSO cross-validation: ", e$message)
    return(Inf)  # Return Inf for failed fits
  })
}

#definded create_model_matrix
create_model_matrix <- function(data, x) {
  data.frame(
    RV = data$RV,
    x_lag1= lag(x,1)
    x_lag5=rollmean(x,5)
    x_lag22=rollmean(x,22)  ) |> na.omit()
}

# another function

# Step 2: Optimize lanta1, lanta2, lanta3
opt_result <- optim(
  par = runif(3),  # Initial values for lanta1, lanta2, lanta3
  fn = loss_function_lanta,
  data = data_get_cj,
  method = "L-BFGS-B",
  lower = c(1e-6, 1e-6, 1e-6),
  upper = c(20, 20, 20),
  control = list(maxit = 1000, trace = 1)
)

# Extract optimized lanta parameters
lanta1_opt <- opt_result$par[1]
lanta2_opt <- opt_result$par[2]
lanta3_opt <- opt_result$par[3]

# Step 3: Final LASSO analysis with optimized lanta and cross-validated lambda
final_lasso_analysis <- function(data, lanta1, lanta2, lanta3) {
  # Calculate measures
  r2t <- sapply(seq_along(data$returns), function(i) 
    conv_fun2_vectorized(TSPL_kernel_vectorized, data$returns[1:i], lanta1))
  
  cjt <- sapply(seq_along(data$returns), function(i) 
    conv_fun1_vectorized(TSPL_kernel_vectorized, data$JV[1:i], lanta2))
  
  cvt <- sapply(seq_along(data$returns), function(i) 
    conv_fun1_vectorized(TSPL_kernel_vectorized, data$CT[1:i], lanta3))
  
  # Create model matrix
  model_data <- create_model_matrix(data, r2t, cjt, cvt)
  X <- as.matrix(model_data[, -1])
  y <- model_data$RV
  
  # Cross-validate and fit final LASSO model
  cv_model <- cv.glmnet(X, y, alpha = 1, nfolds = 15)
  lambda_min <- cv_model$lambda.min
  final_model <- glmnet(X, y, alpha = 1, lambda = lambda_min)
  
  # Extract coefficients and selected variables
  coefficients <- coef(final_model)
  selected_vars <- which(coefficients[-1] != 0)  # Exclude intercept
  variable_names <- colnames(X)
  
  # Calculate performance metrics
  y_pred <- predict(final_model, X)
  mse <- mean((y - y_pred)^2)
  r_squared <- 1 - sum((y - y_pred)^2) / sum((y - mean(y))^2)
  
  results <- list(
    coefficients = coefficients,
    selected_variables = variable_names[selected_vars],
    eliminated_variables = variable_names[-selected_vars],
    performance = list(
      mse = mse,
      r_squared = r_squared,
      lambda = lambda_min
    ),
    model = final_model
  )
  
  # Print summary
  cat("LASSO Regression Results:\n")
  cat("========================\n")
  cat("\nSelected Variables:\n")
  print(results$selected_variables)
  cat("\nModel Performance:\n")
  cat("R-squared:", round(r_squared, 4), "\n")
  cat("MSE:", round(mse, 4), "\n")
  cat("Lambda:", round(lambda_min, 6), "\n")
  
  return(results)
}

# Step 4: Analyze final results
final_results <- final_lasso_analysis(
  data = data_get_cj,
  lanta1 = lanta1_opt,
  lanta2 = lanta2_opt,
  lanta3 = lanta3_opt
)

# Print optimized lanta values
cat("Optimized lanta1:", lanta1_opt, "\n")
cat("Optimized lanta2:", lanta2_opt, "\n")
cat("Optimized lanta3:", lanta3_opt, "\n")
