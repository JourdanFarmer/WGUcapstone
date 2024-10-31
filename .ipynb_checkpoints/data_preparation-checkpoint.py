# data_preparation.py
import pandas as pd

def load_and_clean_data(file_path='olympics-economics.csv'):
    df = pd.read_csv(file_path)

    # Drop rows with missing values
    df = df.dropna()

    # Normalize GDP (using the lowercase column name 'gdp')
    df['gdp_normalized'] = (df['gdp'] - df['gdp'].mean()) / df['gdp'].std()

    # Save the cleaned dataset
    df.to_csv('cleaned_olympics_data.csv', index=False)
    return df

if __name__ == "__main__":
    load_and_clean_data()
