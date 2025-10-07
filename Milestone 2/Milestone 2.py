# Milestone 2 - Product-Level Forecasting (Prophet + LSTM + ARIMA)
# Author: NVS Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# -----------------------------
# 1. Load Cleaned Data
# -----------------------------
df = pd.read_csv("cleaned_retail_sales.csv")
df['date'] = pd.to_datetime(df['date'])

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

print("✅ Data loaded successfully!")
print(df.head())

# -----------------------------
# 2. Helper Functions
# -----------------------------
def train_lstm(series, n_lags=7, epochs=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled) - n_lags):
        X.append(scaled[i:i+n_lags, 0])
        y.append(scaled[i+n_lags, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_lags, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    return model, scaler


def forecast_lstm(model, scaler, series, steps=30, n_lags=7):
    data = scaler.transform(series.values.reshape(-1, 1)).flatten().tolist()
    preds = []
    for _ in range(steps):
        x_input = np.array(data[-n_lags:]).reshape((1, n_lags, 1))
        yhat = model.predict(x_input, verbose=0)
        data.append(yhat[0][0])
        preds.append(yhat[0][0])
    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds


def forecast_arima(series, steps=30):
    try:
        model = ARIMA(series, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast.values
    except Exception as e:
        print("⚠️ ARIMA failed:", e)
        return np.array([np.nan] * steps)


def safe_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# -----------------------------
# 3. Forecasting Loop (Per Product)
# -----------------------------
forecast_list = []
results_summary = []

for product_id in df['product_id'].unique():
    print(f"\n🔄 Training Prophet, LSTM, ARIMA for Product ID: {product_id}...")

    product_df = df[df['product_id'] == product_id][['date', 'units_sold']].copy()
    if len(product_df) < 60:
        print(f"⚠️ Skipping product {product_id} (too few records).")
        continue

    series = product_df.set_index('date')['units_sold']

    # Prophet
    prophet_df = product_df.rename(columns={'date': 'ds', 'units_sold': 'y'})
    model_p = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model_p.fit(prophet_df)
    future = model_p.make_future_dataframe(periods=30)
    forecast_p = model_p.predict(future)
    yhat_p = forecast_p['yhat'][-30:].values

    # LSTM (skip large data)
    if len(series) > 500:
        print(f"⚠️ Skipping LSTM for product {product_id} (too large dataset).")
        yhat_l = np.full(30, np.nan)
        mae_l = rmse_l = mape_l = np.nan
    else:
        train_series = series.iloc[:int(len(series) * 0.8)]
        lstm_model, scaler = train_lstm(train_series, n_lags=5, epochs=5)
        yhat_l = forecast_lstm(lstm_model, scaler, series, steps=30)

    # ARIMA
    yhat_a = forecast_arima(series, steps=30)

    # Evaluation
    actual = series[-30:] if len(series) >= 30 else series

    mae_p = mean_absolute_error(actual, yhat_p[:len(actual)])
    rmse_p = np.sqrt(mean_squared_error(actual, yhat_p[:len(actual)]))
    mape_p = safe_mape(actual, yhat_p[:len(actual)])

    if not np.isnan(yhat_l).all():
        mae_l = mean_absolute_error(actual, yhat_l[:len(actual)])
        rmse_l = np.sqrt(mean_squared_error(actual, yhat_l[:len(actual)]))
        mape_l = safe_mape(actual, yhat_l[:len(actual)])

    mae_a = mean_absolute_error(actual, yhat_a[:len(actual)])
    rmse_a = np.sqrt(mean_squared_error(actual, yhat_a[:len(actual)]))
    mape_a = safe_mape(actual, yhat_a[:len(actual)])

    print(f"Prophet → MAE:{mae_p:.2f} RMSE:{rmse_p:.2f} MAPE:{mape_p:.1f}%")
    print(f"LSTM    → MAE:{mae_l:.2f} RMSE:{rmse_l:.2f} MAPE:{mape_l:.1f}%")
    print(f"ARIMA   → MAE:{mae_a:.2f} RMSE:{rmse_a:.2f} MAPE:{mape_a:.1f}%")

    # Pick best model based on RMSE
    errors = {
        'Prophet': rmse_p,
        'LSTM': rmse_l if not np.isnan(rmse_l) else np.inf,
        'ARIMA': rmse_a
    }
    best_model = min(errors, key=errors.get)
    print(f"🏆 Best Model for Product {product_id}: {best_model}")

    forecast_dates = pd.date_range(start=product_df['date'].max() + pd.Timedelta(days=1), periods=30)
    best_forecast = (
        yhat_p if best_model == "Prophet"
        else yhat_l if best_model == "LSTM"
        else yhat_a
    )

    temp = pd.DataFrame({
        "date": forecast_dates,
        "forecast_best": best_forecast,
        "product_id": product_id,
        "best_model": best_model
    })
    forecast_list.append(temp)

    results_summary.append({
        "product_id": product_id,
        "Prophet_RMSE": rmse_p,
        "LSTM_RMSE": rmse_l,
        "ARIMA_RMSE": rmse_a,
        "Best_Model": best_model
    })

# -----------------------------
# 4. Save Results
# -----------------------------
forecast_all = pd.concat(forecast_list)
forecast_all.to_csv("data/product_forecast_results.csv", index=False)

results_df = pd.DataFrame(results_summary)
results_df.to_csv("data/product_model_performance.csv", index=False)

print("\n✅ Forecasts saved in data/product_forecast_results.csv")
print("✅ Model performance summary saved in data/product_model_performance.csv")

# -----------------------------
# 5. Save Forecast Plots by Product
# -----------------------------
print("\n📊 Generating and saving forecast plots by product...")

for product_id in df['product_id'].unique():
    actual = df[df['product_id'] == product_id]
    forecast_p = forecast_all[forecast_all['product_id'] == product_id]

    if forecast_p.empty:
        continue

    best_model = forecast_p['best_model'].iloc[0]
    plt.figure(figsize=(10, 4))
    plt.plot(actual['date'], actual['units_sold'], label="Actual Sales", color="blue")
    plt.plot(forecast_p['date'], forecast_p['forecast_best'], label=f"Forecast ({best_model})", color="orange")
    plt.title(f"Sales Forecast - Product ID: {product_id} (Best: {best_model})")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/forecast_product_{product_id}.png")
    plt.close()

print("✅ All forecast plots saved in 'plots/' folder.")

# -----------------------------
# 6. Summary Table
# -----------------------------
print("\n📋 Model Summary (Best Model by Product):")
print(results_df[['product_id', 'Best_Model']])
