# modeling.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def kmeans_clustering(file_path='cleaned_olympics_data.csv', n_clusters=3):
    df = pd.read_csv(file_path)
    kmeans = KMeans(n_clusters=n_clusters)
    df['Cluster'] = kmeans.fit_predict(df[['gdp', 'total']])  # Use raw GDP
    df.to_csv('clustered_olympics_data.csv', index=False)
    return df


def linear_regression(file_path='cleaned_olympics_data.csv'):
    df = pd.read_csv(file_path)
    X = df[['gdp']]  # Use raw GDP
    y = df['total']  # Total medals as target variable

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Fit linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    predictions = lin_reg.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Linear Regression Model Mean Squared Error: {mse}")

    return lin_reg, df  # Return both the model and the dataframe


if __name__ == "__main__":
    kmeans_clustering()
    linear_regression()
