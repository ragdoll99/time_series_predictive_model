library(knitr)
library(tidyverse)
library(dplyr)
library(ggplot2)
library(seasonal) 
library(fpp2)
library(forecast)
library(MLmetrics)

df <- read.csv("../data/Q2_Perishable.csv")
head(df)

df$date <- paste("01",df$Month,as.character(df$Year),sep="-")
df$date <- as.Date(df$date, "%d-%b-%Y")
df$date

sales <- ts(df$Sales..in.thousands., frequency = 12, start = 2017)
sales

autoplot(sales, main = "Perishable food items sales: 2017 - 2020")

ggseasonplot(sales, main = "Seasonal Plot")

decomposed_sales_additive <- decompose(sales, type = "additive")
autoplot(decomposed_sales_additive)

decomposed_sales_multiplicative <- decompose(sales, type = "multiplicative")
autoplot(decomposed_sales_multiplicative)

#Create samples
training=window(sales, start = c(2017,1), end = c(2019,12))
test=window(sales, start = c(2020,1))


naive = snaive(training, h=length(test))
Naive_MAPE <- MAPE(naive$mean, test) * 100
Naive_MAPE

plot(sales, col="blue", xlab="Year", ylab="Sales", main="Seasonal Naive Forecast", type='l') +
  lines(naive$mean, col="red", lwd=2)

ets_model = ets(training, allow.multiplicative.trend = TRUE)
summary(ets_model)

ets_forecast = forecast(ets_model, h=length(test))
ETS_MAPE <- MAPE(ets_forecast$mean, test) *100
ETS_MAPE

plot(sales, col="blue", xlab="Year", ylab="Sales", main="ETS Forecast", type='l') +
  lines(ets_forecast$mean, col="red", lwd=2)

dshw_model = dshw(training, period1=3, period2 = 12, h=length(test))
DSHW_MAPE <- MAPE(dshw_model$mean, test)*100
DSHW_MAPE

plot(sales, col="blue", xlab="Year", ylab="Sales", main="DSHW Forecast", type='l') +
  lines(dshw_model$mean, col="red", lwd=2)


tbats_model = tbats(training)
tbats_forecast = forecast(tbats_model, h=length(test))
TBATS_MAPE <- MAPE(tbats_forecast$mean, test) * 100
TBATS_MAPE

plot(sales, col="blue", xlab="Year", ylab="Sales", main="TBATS Forecast", type='l') +
  lines(tbats_forecast$mean, col="red", lwd=2)


arima_optimal = auto.arima(training)
arima_optimal

To forecast using ARIMA model (SARIMA for seasonality), there is a sarima.for function. 

library(astsa)
sarima_forecast = sarima.for(training, n.ahead=12,
                             p=0,d=0,q=0,P=0,D=1,Q=0,S=12, plot=FALSE)
SARIMA_MAPE <- MAPE(sarima_forecast$pred, test) * 100
SARIMA_MAPE

plot(sales, col="blue", xlab="Year", ylab="Sales", main="SARIMA Forecast", type='l') +
  lines(sarima_forecast$pred, col="red", lwd=2)

Model <- c("Naive", "ETS","DSHW", "TBATS","SARIMA")
MAPE <- c(Naive_MAPE, ETS_MAPE, DSHW_MAPE, TBATS_MAPE, SARIMA_MAPE)

c.df <- data.frame(Model, MAPE)
c.df