# jfar205
# Jourdan Farmer
# 011503932

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime

def log_model_performance(file_path='cleaned_olympics_data.csv', log_path='model_performance_log.txt'):
    # Load data
    df = pd.read_csv(file_path)
    X = df[['gdp', 'population']]
    y = df['total']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log performance
    with open(log_path, 'a') as log_file:
        log_file.write(f"{datetime.datetime.now()} - MSE: {mse}, R²: {r2}\n")
        print(f"Model performance logged: MSE = {mse}, R² = {r2}")

if __name__ == "__main__":
    log_model_performance()
