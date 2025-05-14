import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")  # สำหรับเซิร์ฟเวอร์ที่ไม่มี GUI
import matplotlib.pyplot as plt
import io
import base64

class StockPricePredictor:
    def __init__(self, ticker: str, today: str):
        self.ticker = ticker.upper()
        self.today = datetime.strptime(today, "%Y-%m-%d")
        self.start_date = (self.today - timedelta(days=365*10)).strftime("%Y-%m-%d")
        self.scaler = MinMaxScaler()
        self.sequence_length = 60
        self.model = None
        self.data = None
        self.scaled_data = None
        self.X = None
        self.y = None

    def load_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.today.strftime("%Y-%m-%d"))
        if data.empty or len(data) < self.sequence_length:
            raise ValueError(f"Not enough data for {self.ticker} up to {self.today.strftime('%Y-%m-%d')}")
        self.data = data[['Close']]
        self.scaled_data = self.scaler.fit_transform(self.data)

    def prepare_data(self):
        X, y = [], []
        for i in range(self.sequence_length, len(self.scaled_data)):
            X.append(self.scaled_data[i - self.sequence_length:i, 0])
            y.append(self.scaled_data[i, 0])
        self.X = np.array(X).reshape(-1, self.sequence_length, 1)
        self.y = np.array(y)

    def build_model(self):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.sequence_length, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train_model(self, epochs=10, batch_size=32):
        self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict_next_day(self):
        last_60_days = self.scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        predicted_scaled = self.model.predict(last_60_days, verbose=0)
        predicted_price = self.scaler.inverse_transform(predicted_scaled)
        return predicted_price[0][0]

    def plot_prediction_base64(self, predicted_price):
        plt.figure(figsize=(10, 4))
        plt.plot(self.data[-200:], label='Recent Close Prices')
        plt.axhline(y=predicted_price, color='red', linestyle='--', label='Predicted Tomorrow')
        plt.title(f"{self.ticker} Stock Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def run(self):
        self.load_data()
        self.prepare_data()
        self.build_model()
        self.train_model()
        predicted_price = self.predict_next_day()
        plot_base64 = self.plot_prediction_base64(predicted_price)
        return predicted_price, plot_base64