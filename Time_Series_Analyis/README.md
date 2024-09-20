## Perrin Frères Champagne Sales Forecasting

### Overview
In this project, a time series analysis was conducted using the Perrin Frères dataset sourced from Kaggle. The dataset, providing historical sales data, was utilized to develop forecasting models aimed at predicting future sales trends. The key steps and techniques employed in the analysis are outlined below: 


### Data Collection and Preprocessing
During this phase, the dataset was imported from Kaggle.com in CSV format and processed within a Jupyter Notebook environment. During the data preprocessing phase, the sales column was renamed for clarity, and irrelevant rows were removed from the dataset. Subsequently, sales data was aggregated by month to ensure a meaningful granularity for analysis.

### Interpolation Technique
To maintain a continuous timeline for analysis, a full date range spanning from the earliest to the latest dates in the dataset was created with a monthly frequency ('MS'). The DataFrame was then reindexed to align with this complete timeline. Missing sales data points were estimated using linear interpolation to maintain data continuity, and any remaining missing values in the 'sales' column were filled with zeros to complete the dataset.

### Stationarity Check
The Augmented Dickey-Fuller (ADF) test was conducted to assess the stationarity of the sales data

### Differencing: First-Order Differencing
First-order differencing was applied to the sales data to achieve stationarity by computing the difference between consecutive observations at a lag of 12 months, which corresponds to one year. 

### ACF and PACF Plots
Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots were generated and the appropriate values of p and q were determined (p=1 and q=1) for the time series model.
The ACF plot displays the autocorrelation at different lags, while the PACF plot shows the partial autocorrelation at each lag.


### ARIMA Model
An Autoregressive Integrated Moving Average (ARIMA) model was fitted to the sales data. The ARIMA model was specified with parameters (p, d, q) as (1,1,1) respectively. Forecasts for future time periods were generated and visualized, starting from the 91st month and ending at the 105th month.

### SARIMA Model
A Seasonal Autoregressive Integrated Moving Average (SARIMA) model was fitted to the sales data. The SARIMA model was specified with parameters (p, d, q) for the non-seasonal component and (P, D, Q, s) as (1,1,1,12) respectively. Forecasts for future time periods were generated and visualized, starting from the 91st month and ending at the 105th month.

### Results & Discussion
The ADF test yielded a p-value of 0.3639, indicating weak evidence against stationarity. However, after applying first-order differencing, the p-value decreased significantly to 0.0000, indicating that the differenced data is stationary. ACF and PACF plots aided in determining the appropriate values of p and q for time series modeling. Comparing the forecasts generated by the ARIMA and SARIMA models, the SARIMA model provided more accurate forecasts, highlighting the importance of accounting for seasonal patterns in time series modeling.

### Conclusion
The combination of stationarity checks, differencing, and analysis of ACF and PACF plots allows for the identification of appropriate parameters for time series modeling. Additionally, the comparison between ARIMA and SARIMA models demonstrates the importance of selecting the most appropriate model for the specific characteristics of the data, especially when dealing with seasonal patterns. 
This project holds significant value for businesses and business owners to optimize operations and marketing strategies, and also to minimize costs and plan for long-term. 
It's important to acknowledge that while this serves as a simplified illustration, real-world applications often entail more intricate data preprocessing and feature engineering methodologies.