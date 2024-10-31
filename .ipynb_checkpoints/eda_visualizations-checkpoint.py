# eda_visualizations.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_gdp_vs_medals(file_path='cleaned_olympics_data.csv'):
    df = pd.read_csv(file_path)
    sns.scatterplot(data=df, x='gdp_normalized', y='total')
    plt.title('GDP vs. Total Olympic Medals')
    plt.xlabel('Normalized GDP')
    plt.ylabel('Total Medals')
    plt.show()

if __name__ == "__main__":
    plot_gdp_vs_medals()
