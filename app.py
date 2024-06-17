import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px

#Model Loading
# Load the model with specific options for compatibility
model_path = r"C:\Users\umang\JupyterLab\Bitcoin_Price_Prediction_model.keras"
model = keras.models.load_model(model_path, compile=False)  # Disable compilation if not needed

# Example of adjusting input shape for older models (if needed)
if isinstance(model.layers[0], keras.layers.InputLayer):
    model.layers[0].batch_input_shape = (None, 100, 1)  # Adjust input shape if necessary

# Compile the model if needed
model.compile(optimizer='adam', loss='mse')


# Webpage Headers - Centered and Styled
st.write(
    "<h1 style='text-align: center; color: #8a2be2;'>BitPredict</h1>", 
    unsafe_allow_html=True
)
st.markdown("<h3 style='text-align: center; color: #F1A5AD; font-size: smaller;'>Bitcoin Prediction Model</h3>", unsafe_allow_html=True)
st.markdown("---")  # Adding a horizontal line for separation

st.markdown("<h3 style='text-align: center;'>Bitcoin Price Data</h3>", unsafe_allow_html=True)

#Data Fetch
# Calculate end_date as today's date minus one day
end_date = datetime.today() - timedelta(days=1)
data = pd.DataFrame(yf.download('BTC-USD', '2015-01-01', end_date))
data = data.reset_index()
st.write(data)
# Add space after Bitcoin Price Data section
st.markdown("---")


#Plotting Trend Chart - with date references on x-axis
st.markdown("<h3 style='text-align: center;'>Bitcoin Trend Chart</h3>", unsafe_allow_html=True)
price_data = data[['Date', 'Close']]
price_data['Date'] = price_data['Date'].astype(str)
price_data.set_index('Date', inplace=True)
st.line_chart(price_data['Close'])
# Add space after Bitcoin Trend Chart section
st.markdown("---")


#Training and Testing data
train_data = data[:-700]   #Use all except last 600 days data to train
test_data = data[-700:]    #Use last 600 days data to test

#Scaling the data to plot but only scaling close values(numeric values only)
scaler = MinMaxScaler(feature_range=(0,1))
train_data_scale = scaler.fit_transform(train_data[['Close']])
test_data_scale = scaler.transform(test_data[['Close']])

#Creating dataset for predictions
base_days = 350
x = []
y = []
for i in range(base_days, test_data_scale.shape[0]):
    x.append(test_data_scale[i-base_days:i])
    y.append(test_data_scale[i,0])

x, y  = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

#Predictions
st.markdown("<h3 style='text-align: center;'>Predicted v/s Actual Price</h3>", unsafe_allow_html=True)

pred = model.predict(x)
pred = scaler.inverse_transform(pred)
preds = pred.reshape(-1, 1)
ys = scaler.inverse_transform(y.reshape(-1, 1))

# Calculate Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(ys, preds)

# Calculate Weighted Mean Absolute Percentage Error (WMAPE)
wmape = np.mean(np.abs((ys - preds) / ys)) * 100

# Calculate Symmetric Mean Absolute Percentage Error (sMAPE)
smape = np.mean(2 * np.abs(preds - ys) / (np.abs(ys) + np.abs(preds))) * 100

# Display MAPE, WMAPE, and sMAPE in percentage
st.markdown("<p style='text-align: center; font-size: smaller;'>Weighted Mean Absolute Percentage Error (WMAPE): <b>{:.2f}%</b></p>".format(wmape), unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: smaller;'>Symmetric Mean Absolute Percentage Error (sMAPE): <b>{:.2f}%</b></p>".format(smape), unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: smaller;'>MAPE: {:.2f}%</p>".format(mape), unsafe_allow_html=True)
#Create dataset for Comparision of both the prices
preds = pd.DataFrame(preds, columns=['Predicted Price'])
ys = pd.DataFrame(ys, columns=['Actual Price'])
dates = price_data.index[-len(preds):]
comparison_df = pd.DataFrame({'Date': dates, 'Predicted Price': preds['Predicted Price'], 'Actual Price': ys['Actual Price']})
comparison_df['Date'] = comparison_df['Date'].astype(str)

#Plotting the prediction accuray chart using Plotly
fig = px.line(comparison_df, x='Date', y=['Predicted Price', 'Actual Price'],
              labels={'value': 'Price', 'Date': 'Date'}, title='Prediction Accuracy Chart - Predicted vs Actual Price')
fig.update_traces(line=dict(color='yellow'), selector=dict(name='Predicted Price'))
fig.update_traces(line=dict(color='green'), selector=dict(name='Actual Price'))
st.plotly_chart(fig)
# Add space after Prediction Accuracy Chart section
st.markdown("---")

#Future Predictions
m = y
z = []
future_days = 30
for i in range(base_days, len(m) + future_days):
    m = m.reshape(-1, 1)
    inter = [m[-base_days:, 0]]
    inter = np.array(inter)
    inter = np.reshape(inter, (inter.shape[0], inter.shape[1], 1))
    pred = model.predict(inter)
    m = np.append(m, pred)
    z = np.append(z, pred)

#Transform future predictions back to original scale
z = np.array(z)
z = scaler.inverse_transform(z.reshape(-1, 1))

#Display future predictions
st.markdown("<h3 style='text-align: center;'>Next 1 month Price Prediction</h3>", unsafe_allow_html=True)
future_dates = pd.date_range(start=price_data.index[-1], periods=future_days + 1)[1:]
future_pred_df = pd.DataFrame(z, index=future_dates, columns=['Predicted Price'])
fig = px.line(future_pred_df, x=future_dates, y='Predicted Price')
fig.update_layout(xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'))
fig.update_traces(line=dict(color='deeppink', width=2))  # Adjust line color to neon green
st.plotly_chart(fig)
# Add space after Next 1 month Price Prediction section
st.markdown("---")

#Display future 10 days predictions in a table
st.markdown("<h3 style='text-align: center;'>Next 10 days Predicted Prices</h3>", unsafe_allow_html=True)
future_dates_formatted = [date.strftime('%b %d') for date in future_dates[:10]]  # Format dates as month day
future_predictions_df = pd.DataFrame(z[:10], index=future_dates_formatted, columns=['Predicted Price'])
future_predictions_df.index.name = 'Date'
st.table(future_predictions_df)
# Add space after Next 10 days Predicted Prices section
st.markdown("---")


# Developer Information
st.markdown("<p style='text-align: center; font-size: smaller; color: #01FFF4; font-weight: bold;'>Developed by - Umang Raj  ||  Copyright@2024 BitPredict</p>", unsafe_allow_html=True)
