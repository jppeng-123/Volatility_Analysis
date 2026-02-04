# -*- coding: utf-8 -*-
""" Bitcoin GARCH & EGARCH Vol Model 

"""

import gdown
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm


warnings.filterwarnings("ignore")

# Download the Bitcoin price data
url = 'https://drive.google.com/uc?id=1bJVxqJN4jQLpc8dgR0W9PVRms1LV5O2d'
output = 'Bitcoin_Price.csv'
gdown.download(url, output, quiet=False)

# Load the Bitcoin price data, skip the first 6 rows (metadata)
btc_garch = pd.read_csv(output, skiprows=6)

# Rename the columns for clarity
btc_garch.columns = ['Date', 'Close']

# Convert the 'Date' column to datetime format, ensuring we skip errors
btc_garch['Date'] = pd.to_datetime(btc_garch['Date'], errors='coerce')

# Remove any rows with invalid dates (NaT) or missing values
btc_garch = btc_garch.dropna().sort_values('Date')
btc_garch.reset_index(drop=True, inplace=True)

# Calculate log returns
btc_garch['Log_Returns'] = btc_garch['Close'].pct_change().apply(lambda x: np.log(1 + x)).dropna()

# Fit the GARCH(1,1) model
garch_model = arch_model(btc_garch['Log_Returns'].dropna(), vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(disp='off')

# Plot the conditional volatility with Date on x-axis
volatility = garch_fit.conditional_volatility

# Plot the volatility with Date as x-axis
plt.figure(figsize=(15, 6))
plt.plot(btc_garch['Date'].iloc[1:], volatility)
plt.title('GARCH(1,1) Conditional Volatility for Bitcoin')
plt.xlabel('Date')
plt.ylabel('Conditional Volatility')
plt.xticks(rotation=45)
plt.show()
warnings.filterwarnings("ignore")

# Calculate the 30-day rolling volatility
btc_garch['Rolling_Volatility'] = btc_garch['Log_Returns'].rolling(window=15).std()

# Plot GARCH(1,1) Conditional Volatility and 30-day Rolling Volatility
plt.figure(figsize=(15, 6))
plt.plot(btc_garch['Date'].iloc[1:], volatility, label='GARCH(1,1) Conditional Volatility')
plt.plot(btc_garch['Date'], btc_garch['Rolling_Volatility'], label='15-Day Rolling Volatility', color='orange')
plt.title('GARCH(1,1) Conditional Volatility and 15-Day Rolling Volatility for Bitcoin')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.xticks(rotation=45)
plt.show()



# Drop NaN values from the actual (rolling) and predicted (GARCH) volatility series
valid_data = btc_garch.dropna(subset=['Rolling_Volatility'])
garch_volatility = volatility[valid_data.index[1:]]  # Match GARCH series to valid_data length
actual_volatility = valid_data['Rolling_Volatility'].iloc[1:]  # Match index

# Calculate R^2, MSE, RMSE, MAE, MAPE
r2 = r2_score(actual_volatility, garch_volatility)
mse = mean_squared_error(actual_volatility, garch_volatility)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_volatility, garch_volatility)
mape = np.mean(np.abs((actual_volatility - garch_volatility) / actual_volatility)) * 100

# Calculate p-value using regression
X = sm.add_constant(garch_volatility)
model = sm.OLS(actual_volatility, X).fit()
p_value = model.pvalues[1]  # p-value of the GARCH volatility term

# Display all results
print(f"R^2: {r2:.10f}")
print(f"MSE: {mse:.10f}")
print(f"RMSE: {rmse:.10f}")
print(f"MAE: {mae:.10f}")
print(f"MAPE: {mape:.10f}%")
print(f"P-value: {p_value:.10f}")
warnings.filterwarnings("ignore")

btc_egarch = pd.read_csv(output, skiprows=6)
btc_egarch.columns = ['Date', 'Close']
btc_egarch['Date'] = pd.to_datetime(btc_egarch['Date'], errors='coerce')
btc_egarch = btc_egarch.dropna().sort_values('Date')
btc_egarch.reset_index(drop=True, inplace=True)
btc_egarch['Log_Returns'] = btc_egarch['Close'].pct_change().apply(lambda x: np.log(1 + x)).dropna()

egarch_model = arch_model(btc_egarch['Log_Returns'].dropna(), vol='EGarch', p=1, q=1)
egarch_fit = egarch_model.fit(disp='off')

print(egarch_fit.summary())

volatility = egarch_fit.conditional_volatility

plt.figure(figsize=(15, 6))
plt.plot(btc_egarch['Date'].iloc[1:], volatility)
plt.title('EGARCH(1,1) Conditional Volatility for Bitcoin')
plt.xlabel('Date')
plt.ylabel('Conditional Volatility')
plt.xticks(rotation=45)
plt.show()
warnings.filterwarnings("ignore")

# Calculate the 30-day rolling volatility
btc_egarch['Rolling_Volatility'] = btc_egarch['Log_Returns'].rolling(window=15).std()

# Plot GARCH(1,1) Conditional Volatility and 30-day Rolling Volatility
plt.figure(figsize=(15, 6))
plt.plot(btc_egarch['Date'].iloc[1:], volatility, label='EGARCH(1,1) Conditional Volatility')
plt.plot(btc_egarch['Date'], btc_egarch['Rolling_Volatility'], label='15-Day Rolling Volatility', color='orange')
plt.title('EGARCH(1,1) Conditional Volatility and 15-Day Rolling Volatility for Bitcoin')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.xticks(rotation=45)
plt.show()


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import numpy as np

# Drop NaN values from the actual (rolling) and predicted (EGARCH) volatility series
valid_data = btc_egarch.dropna(subset=['Rolling_Volatility'])
egarch_volatility = volatility[valid_data.index[1:]]  # Match EGARCH series to valid_data length
actual_volatility = valid_data['Rolling_Volatility'].iloc[1:]  # Match index

# Calculate R^2, MSE, RMSE, MAE, MAPE
e_r2 = r2_score(actual_volatility, egarch_volatility)
e_mse = mean_squared_error(actual_volatility, egarch_volatility)
e_rmse = np.sqrt(mse)
e_mae = mean_absolute_error(actual_volatility, egarch_volatility)
e_mape = np.mean(np.abs((actual_volatility - egarch_volatility) / actual_volatility)) * 100

# Calculate p-value using regression
X = sm.add_constant(egarch_volatility)
model = sm.OLS(actual_volatility, X).fit()
e_p_value = model.pvalues[1]  # p-value of the EGARCH volatility term

# Display all results
print(f"R^2: {e_r2:.10f}")
print(f"MSE: {e_mse:.10f}")
print(f"RMSE: {e_rmse:.10f}")
print(f"MAE: {e_mae:.10f}")
print(f"MAPE: {e_mape:.10f}%")
print(f"P-value: {e_p_value:.10f}")
warnings.filterwarnings("ignore")

# Create a combined plot for GARCH and EGARCH
plt.figure(figsize=(15, 6))

# Plot GARCH volatility
plt.plot(btc_garch['Date'].iloc[1:], garch_fit.conditional_volatility, color = 'blue', label='GARCH(1,1)')

# Plot EGARCH volatility
plt.plot(btc_egarch['Date'].iloc[1:], egarch_fit.conditional_volatility,color = 'red', label='EGARCH(1,1)')

# Add titles, labels, and legend
plt.title('GARCH(1,1) vs EGARCH(1,1) Conditional Volatility for Bitcoin')
plt.xlabel('Date')
plt.ylabel('Conditional Volatility')
plt.xticks(rotation=45)
plt.legend()

# Display the combined plot
plt.show()
warnings.filterwarnings("ignore")

import gdown
import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


url1 = 'https://drive.google.com/uc?id=1RrH0FnkD3WRpa6z8-8R6ifFBR18HbbUf'
output1 = 'btc_usd_q3.csv'
gdown.download(url1, output1, quiet=False)

btc_holdout = pd.read_csv(output1, skiprows=6)
btc_holdout.columns = ['Date', 'Close']
btc_holdout['Date'] = pd.to_datetime(btc_holdout['Date'], errors='coerce')
btc_holdout = btc_holdout.dropna().sort_values('Date')
btc_holdout.reset_index(drop=True, inplace=True)


btc_holdout['Log_Returns'] = btc_holdout['Close'].pct_change().apply(lambda x: np.log(1 + x))
btc_holdout = btc_holdout.dropna(subset=['Log_Returns'])
btc_holdout['Rolling_Volatility'] = btc_holdout['Log_Returns'].rolling(window=15).std()
btc_holdout = btc_holdout.dropna(subset=['Rolling_Volatility'])
btc_holdout.reset_index(drop=True, inplace=True)

garch_forecast_volatility = []
egarch_forecast_volatility = []


for i in range(len(btc_holdout)):
    train_data = pd.concat([btc_garch['Log_Returns'], btc_holdout['Log_Returns'][:i]]).dropna()


    garch_model = arch_model(train_data, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp="off")
    garch_forecast = garch_fit.forecast(horizon=1)
    garch_forecast_volatility.append(np.sqrt(garch_forecast.variance.values[-1, 0]))


    egarch_model = arch_model(train_data, vol='EGarch', p=1, q=1)
    egarch_fit = egarch_model.fit(disp="off")
    egarch_forecast = egarch_fit.forecast(horizon=1)
    egarch_forecast_volatility.append(np.sqrt(egarch_forecast.variance.values[-1, 0]))


garch_forecast_volatility = np.array(garch_forecast_volatility)
egarch_forecast_volatility = np.array(egarch_forecast_volatility)


plt.figure(figsize=(15, 6))
plt.plot(btc_holdout['Date'], btc_holdout['Rolling_Volatility'], label='Actual 15-Day Rolling Volatility')
plt.plot(btc_holdout['Date'], garch_forecast_volatility, label='GARCH(1,1) Forecasted Volatility')
plt.plot(btc_holdout['Date'], egarch_forecast_volatility, label='EGARCH(1,1) Forecasted Volatility', color='orange')
plt.title('Holdout Test: Actual(Rolling 15-Day) vs. Forecasted Volatility for Q3 2024')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.xticks(rotation=45)
plt.show()

garch_r2 = r2_score(btc_holdout['Rolling_Volatility'], garch_forecast_volatility)
garch_mse = mean_squared_error(btc_holdout['Rolling_Volatility'], garch_forecast_volatility)
garch_rmse = np.sqrt(garch_mse)
garch_mae = mean_absolute_error(btc_holdout['Rolling_Volatility'], garch_forecast_volatility)
garch_mape = np.mean(np.abs((btc_holdout['Rolling_Volatility'] - garch_forecast_volatility) / btc_holdout['Rolling_Volatility'])) * 100


print(f"GARCH(1,1) Model Performance on Holdout Data:")
print(f"R^2: {garch_r2:.10f}")
print(f"MSE: {garch_mse:.10f}")
print(f"RMSE: {garch_rmse:.10f}")
print(f"MAE: {garch_mae:.10f}")
print(f"MAPE: {garch_mape:.10f}%")

egarch_r2 = r2_score(btc_holdout['Rolling_Volatility'], egarch_forecast_volatility)
egarch_mse = mean_squared_error(btc_holdout['Rolling_Volatility'], egarch_forecast_volatility)
egarch_rmse = np.sqrt(egarch_mse)
egarch_mae = mean_absolute_error(btc_holdout['Rolling_Volatility'], egarch_forecast_volatility)
egarch_mape = np.mean(np.abs((btc_holdout['Rolling_Volatility'] - egarch_forecast_volatility) / btc_holdout['Rolling_Volatility'])) * 100


print(f"\nEGARCH(1,1) Model Performance on Holdout Data:")
print(f"R^2: {egarch_r2:.10f}")
print(f"MSE: {egarch_mse:.10f}")
print(f"RMSE: {egarch_rmse:.10f}")
print(f"MAE: {egarch_mae:.10f}")
print(f"MAPE: {egarch_mape:.10f}%")
warnings.filterwarnings("ignore")
