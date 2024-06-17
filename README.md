This project makes use of the following Python Lib:
Note: This project is for educational and informational purposes only. Cryptocurrency investments are highly volatile, and predictions should not be considered financial advice.

# BitPredict: Bitcoin Price Prediction Model

Welcome to **BitPredict**, a real-time Bitcoin price prediction model that leverages machine learning to provide accurate and up-to-date predictions. This repository contains the code for a Streamlit web application that fetches real-time Bitcoin price data, processes it, and predicts future prices.

## Features

- **Real-Time Data Fetching**: Fetches real-time Bitcoin price data from Yahoo Finance.
- **Machine Learning Model**: Uses a pre-trained TensorFlow model to predict future Bitcoin prices.
- **Interactive Visualizations**: Displays historical price trends and prediction accuracy using Plotly.
- **Future Predictions**: Provides price predictions for the next 30 days.
- **User-Friendly Interface**: Interactive Streamlit dashboard for easy use.

## Project Structure
- app.py: Main Streamlit application script.
- requirements.txt: Python dependencies required for the project.

## Usage
- Bitcoin Price Data: View real-time Bitcoin price data fetched from Yahoo Finance.
- Bitcoin Trend Chart: Visualize historical price trends.
- Prediction Accuracy Chart: Compare predicted prices with actual prices.
- Next 1 Month Price Prediction: View predicted prices for the next 30 days.
- Next 10 Days Predicted Prices: View a table of predicted prices for the next 10 days.

## Model Details
- The model is trained using historical Bitcoin price data.
- It uses LSTM (Long Short-Term Memory) neural networks for time series prediction.
- The training data includes Bitcoin prices from January 2015 to the present day minus one day.
- The model evaluates performance using metrics like MAPE (Mean Absolute Percentage Error), WMAPE (Weighted MAPE), and sMAPE (Symmetric MAPE).

## Developer Information
Developed by - Umang Raj
©2024 BitPredict

### Prerequisites

- Python 3.7+
- pip (Python package installer)

**Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/BitPredict.git
   cd BitPredict
Ensure you have the pre-trained TensorFlow model saved at C:\Users\umang\Bitcoin_Price_Prediction_model.keras. Adjust the path in the code if your model is saved elsewhere.
