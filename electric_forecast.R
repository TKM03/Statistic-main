library(forecast)
library(tseries)

#Step 0
data <- read.csv("Electric_Production.csv")
data()

#plot data 
data <- ts(data[,2], frequency = 12, start = c(1985,1))
plot(data, main = "Time Series Plot", xlab = "Time", ylab = "Value")

#train test split
train <- head(data, round(length(data) * 0.70))
h <- length(data) - length(train)
test <- tail(data, h)
train
test
autoplot(train) + autolayer(test)

#Step 2
#transformation
data_transform <- log(data)
data_transform <- ts(data_transform, frequency = 12, start = c(1985,1))
plot(data_transform, main = "Time Series Plot (Log Transform)", xlab = "Time", ylab = "Value")

#Step 3a Stationarize the Series
#Differencing
# First-order differencing
diff_data <- diff(data_transform)

# Plot differenced data
plot(diff_data, main = "Differenced Time Series", xlab = "Time", ylab = "Differences")

#Step 3b Check the stationarity of the series
adf.test(diff_data)

#Step 4
#auto arima model
arima_model <- auto.arima(train)
summary(arima_model)

#residuals of auto arima model
residuals <- residuals(arima_model)
residuals

#acf of the residuals
acf_residuals <-acf(residuals, main="ACF of Residuals")
ACF$acf_residuals


#ACF
acf <- acf(train, main="Correlogram for the Electric Dataset")
acf$acf

#Version 2
#=================================================================================================================================================================================================#

# Load required libraries
library(forecast)
library(tseries)
library(ggplot2)

# Load data
data <- read.csv("Electric_Production.csv")
data()

# Convert to time series (assuming column 2 is the value column)
data_ts <- ts(data[,2], frequency = 12, start = c(1985,1))
plot(data_ts, main = "Time Series Plot", xlab = "Time", ylab = "Electricity Production")

# Train-test split (70% train, 30% test)
train_length <- round(length(data_ts) * 0.70)
train <- head(data_ts, train_length)
test <- tail(data_ts, length(data_ts) - train_length)
cat("Train length:", length(train), "\nTest length:", length(test), "\n")

# Visualize train and test
autoplot(train, series="Train") + 
  autolayer(test, series="Test") +
  ggtitle("Train and Test Split") +
  xlab("Time") + ylab("Electricity Production") +
  theme_minimal()

# Log transformation
data_transform <- log(data_ts)
data_transform_ts <- ts(data_transform, frequency = 12, start = c(1985,1))
plot(data_transform_ts, main = "Time Series Plot (Log Transform)", xlab = "Time", ylab = "Log Value")

# Differencing
diff_data <- diff(data_transform_ts)
plot(diff_data, main = "Differenced Time Series", xlab = "Time", ylab = "Differences")

# Stationarity test
adf_result <- adf.test(diff_data)
print(adf_result)  # Check p-value < 0.05 for stationarity

# Fit ARIMA model on training data
arima_model <- auto.arima(train)
summary(arima_model)

# Residuals analysis
residuals <- residuals(arima_model)
autoplot(residuals, main="Residuals of ARIMA Model") + 
  xlab("Time") + ylab("Residuals")

# ACF of residuals
acf_residuals <- acf(residuals, main="ACF of Residuals")
print(acf_residuals$acf)  # Access ACF values

# PACF of residuals (additional diagnostic)
pacf(residuals, main="PACF of Residuals")

# Ljung-Box test for residual autocorrelation
box_test <- Box.test(residuals, lag=12, type="Ljung-Box")
print(box_test)  # p-value > 0.05 indicates no significant autocorrelation

# ACF of original train data
acf_train <- acf(train, main="Correlogram for the Electric Dataset")
print(acf_train$acf)

# Forecast for test period
forecast_horizon <- length(test)
forecast <- forecast(arima_model, h=forecast_horizon)
autoplot(forecast) + 
  autolayer(test, series="Actual Test") +
  ggtitle("ARIMA Forecast vs Actual Test Data") +
  xlab("Time") + ylab("Electricity Production") +
  theme_minimal()

# Evaluate forecast accuracy
accuracy_measures <- accuracy(forecast, test)
print(accuracy_measures)  # RMSE, MAE, etc. for train and test
