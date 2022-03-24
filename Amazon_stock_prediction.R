#Getting AMAZON stock dataset and loading the needed packages
if(!require(quantmod)) install.packages("quantmod")
if(!require(forecast)) install.packages("forecast")
if(!require(xlsx)) install.packages("xlsx")
if(!require(tseries)) install.packages("tseries")
if(!require(timeSeries)) install.packages("timeSeries")
if(!require(dplyr)) install.packages("dplyr")
if(!require(fGarch)) install.packages("fGarch")
if(!require(prophet)) install.packages("prophet")
library(prophet)
library(quantmod)
library(forecast)
library("xlsx")
library(tseries)
library(timeSeries)
library(dplyr)
library(fGarch)

getSymbols("AMZN", src="yahoo", from="2015-01-01")


# Conduct ADF test for dataset
print(adf.test(close_price))

#We apply auto arima to the dataset 
modelfit <- auto.arima(close_price, lambda = "auto")

#Box test for lag=2
Box.test(modelfit$residuals, lag= 2, type="Ljung-Box")


#Box test for lag=2
Box.test(modelfit$residuals, type="Ljung-Box")

#Dataset forecasting for the next 30 days
price_forecast <- forecast(modelfit, h=30)

#Dataset forecasting plot for the next 30 days
plot(price_forecast)



#Dataset forecast mean first 5 values
head(price_forecast$mean)



#Dataset forecast lower first 5 values
head(price_forecast$lower)

#Dataset forecast upper first 5 values
head(price_forecast$upper)


#Dividing the data into train and test, applying the model
N = length(close_price)
n = 0.7*N
train = close_price[1:n, ]
test  = close_price[(n+1):N,  ]
trainarimafit <- auto.arima(train, lambda = "auto")
predlen=length(test)
trainarimafit <- forecast(trainarimafit, h=predlen)

#Plotting mean predicted values vs real data
meanvalues <- as.vector(trainarimafit$mean)
precios <- as.vector(test$AMZN.Close)
plot(meanvalues, type= "l", col= "red")
lines(precios, type = "l")



#Dataset forecast upper first 5 values
if(!require(rugarch)) install.packages("rugarch")


library(rugarch)

#Dataset forecast upper first 5 values
fitarfima = autoarfima(data = close_price, ar.max = 2, ma.max = 2, 
                       criterion = "AIC", method = "full")


#define the model
garch11closeprice=ugarchspec(variance.model=list(garchOrder=c(1,1)), mean.model=list(armaOrder=c(1,2)))
#estimate model 
garch11closepricefit=ugarchfit(spec=garch11closeprice, data=close_price)



#conditional volatility plot
plot.ts(sigma(garch11closepricefit), ylab="sigma(t)", col="blue")



#Model akike
infocriteria(garch11closepricefit)


#Normal residuals
garchres <- data.frame(residuals(garch11closepricefit))  
plot(garchres$residuals.garch11closepricefit.)


#Standardized residuals
garchres <- data.frame(residuals(garch11closepricefit, standardize=TRUE)) 
#Normal Q plot
qqnorm(garchres$residuals.garch11closepricefit..standardize...TRUE.)
qqline(garchres$residuals.garch11closepricefit..standardize...TRUE.)


#Squared standardized residuals Ljung Box
garchres <- data.frame(residuals(garch11closepricefit, standardize=TRUE)^2) 
Box.test(garchres$residuals.garch11closepricefit..standardize...TRUE..2, type="Ljung-Box")


#GARCH Forecasting
garchforecast <- ugarchforecast(garch11closepricefit, n.ahead = 30 )

#We convert dataset as prophet input requires
df <- data.frame(ds = index(AMZN),
                 y = as.numeric(AMZN[,'AMZN.Close']))



#prophet model application
prophetpred <- prophet(df)
future <- make_future_dataframe(prophetpred, periods = 30)
forecastprophet <- predict(prophetpred, future)


#Creating train prediction datset to compare the real data
dataprediction <- data.frame(forecastprophet$ds,forecastprophet$yhat)
trainlen <- length(close_price)
dataprediction <- dataprediction[c(1:trainlen),]


#Creating cross validation 
accuracy(dataprediction$forecastprophet.yhat,df$y)


#Creating cross validation 
prophet_plot_components(prophetpred,forecastprophet)


#Loading time series forecasting nearest neighbors package
if(!require(tsfknn)) install.packages("tsfknn")

library(tsfknn)

#Dataframe creation and model application
df <- data.frame(ds = index(AMZN),
                 y = as.numeric(AMZN[,'AMZN.Close']))

predknn <- knn_forecasting(df$y, h = 30, lags = 1:30, k = 40, msas = "MIMO")


#Train set model accuracy
ro <- rolling_origin(predknn)

print(ro$global_accu)

#Hidden layers creation
alpha <- 1.5^(-10)
hn <- length(close_price)/(alpha*(length(close_price)+30))

#Fitting nnetar
lambda <- BoxCox.lambda(close_price)
dnn_pred <- nnetar(close_price, size= hn, lambda = lambda)


#Fitting nnetar
dnn_forecast <- forecast(dnn_pred, h= 30, PI = TRUE)
plot(dnn_forecast)


